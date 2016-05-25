package NaiveBayes

import NaiveBayes.NaiveBayesSupportMethods._
import edu.arizona.sista.processors.Document
import edu.arizona.sista.struct.{Counter, Lexicon}
import Utils.PreProcessing._

/**
  * `Class` to train a Naive Bayes model
  * @param docs The documents to be used in training
  * @param labels The labels for the [[docs]]
  * @param stopWords Any **lowercased** `stop words` to be filtered out of the calculations
  * @param gram `word`s or `lemma`s to be used
  * @param n Size of n-gram
  * @param bernoulli If `true`, then counts will be indicators(0 or 1); otherwise, counts will be frequency counts
  * @param smoothingAlpha if `1d` then "Laplace" smoothing, else if `<1d` then Lidstone smoothing
  * @param documentFrequencyThreshold Minimum percentage of documents in which a token must be present to be included in calculations <br>
  *                                     *.i.e* `documentFrequencyThreshold = .5` means that only tokens that appear in more than half of the training documents will be calculated
  * @param mutualInformationThreshold
  * @param informationGainThreshold
  * @todo Implement feature selection
  * @see [http://www.surdeanu.info/mihai/teaching/ista555-spring15/readings/yang97comparative.pdf]
  * @todo Add capacity to serialize import pieces needed for testing
  * @todo Add capacity to add training documents without completely retraining
  */
class NaiveBayesTraining(
                          val docs: Vector[Document],
                          val labels: Vector[Int],
                          val stopWords: Set[String] = Set(),
                          val gram: String = "word", //"word" or "lemma"
                          val n: Int = 1,
                          val bernoulli: Boolean = false, //http://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html
                          val smoothingAlpha: Double = 1d, //if `1d` then Laplace, if less than `1d` then Lidstone
                          var documentFrequencyThreshold: Double = 0,
                          var mutualInformationThreshold: Int = 0,
                          var informationGainThreshold: Int = 0
                        ) {

  /**
    * Total number of documents in training
    */
  val numberOfDocs = getNumberOfDocs

  /**
    * Number of documents in each class
    */
  val classCounter = getClassCounts

  /**
    * All possible classes
    */
  val possibleClasses = this.classCounter.keySet

  /**
    * Calculated *class priors*
    */
  val classPriors = getClassPriors

  /**
    * [[Lexicon]] of entire training vocabulary (with stop words filtered out) <br>
    * Count of how many documents each token appears in <br>
    * Raw counts of tokens for each class
    */
  var (lex, documentCounter, tokenCounter) = if (this.bernoulli) {
                                extractVocabulary(bernoulli = true, gram = this.gram, n = this.n)
                              } else {
                                extractVocabulary(bernoulli = false, gram = this.gram, n = this.n)
                              }

  /**
    * The denominators for *conditional probability* calculations <br>
    *   `multinomial`: `P(t|c) = (count of token in class + 1) / (number of all tokens in class + size of vocabulary)`
    *   `bernoulli`: `P(t|c) = (number of documents containing token + 1) / (total number of documents in class + 2)`
    */
  val condProbDenominators = (for (c <- this.possibleClasses) yield {
                                if (this.bernoulli) {
                                  c -> this.calculateBernoulliCondProbDenominator(c)
                                } else {
                                  c -> this.calculateMultiCondProbDenominator(c)
                                }
                              }).toMap


  /**
    * *Conditional probabilities* for all tokens and all classes
    */
  val condProbCounters = this.getCondProbs


  /**
    * Get number of documents in collection
    */
  def getNumberOfDocs: Double = {
    this.docs.length.toDouble
  }


  /**
    * Get class counts
    */
  def getClassCounts: Counter[Int] = {

    val counter = new Counter[Int]()

    this.labels.foreach(counter.incrementCount(_))

    counter
  }


  /**
    * Get class *prior probabilities* <br>
    *
    * @return `P(c) = number of documents in class / total number of documents`
    */
  def getClassPriors: Counter[Int] = {

    val counter = new Counter[Int]()

    this.possibleClasses.foreach(z =>
      counter.setCount(z, this.classCounter.getCount(z) / this.numberOfDocs))

    counter
  }


  /**
    * Gets vocabulary from all documents and builds class counters
    *
    * @param bernoulli If `true`, counter contains indicator counts (1 or 0); if `false`, counter contains raw frequency counts
    * @param gram `word` or `lemma`
    * @param n Size of n-gram
    */
  def extractVocabulary(bernoulli: Boolean, gram: String, n: Int = 1): (Lexicon[String], Counter[String], Map[Int, Counter[String]]) = {

    //make a Map of Counters with keys of each label
    val counterMap = (for (c <- this.possibleClasses) yield {
      c -> new Counter[String]()
    }).toMap

    //make a counter for document counts of each token
    val documentCounter = new Counter[String]()

    //make a lexicon
    val lex = new Lexicon[String]()

    //generate one counter for each class
      //concatenate all documents of that class into the one counter
    for (d <- this.docs.zip(this.labels)) {

      //lex specific to this document
        //used by bernoulli and for documentFrequency feature selection
      val documentLex = new Lexicon[String]()

      //split tuple
      val doc = d._1
      val label = d._2

      //initialize variable to house tokens (will be "word" or "lemma")
      var tokensRaw = Vector[String]()

      //iterate through sentences of doc
      for (sentence <- doc.sentences) {

        //get word or lemma and filter stop words
        gram match {
          case "word" => tokensRaw = sentence.words.toVector.
                                      map(word => word.toLowerCase).
                                      filterNot(word => this.stopWords.contains(word))
          case "lemma" => tokensRaw = sentence.lemmas.get.toVector.
                                      map(lemma => lemma.toLowerCase).
                                      filterNot(lemma => this.stopWords.contains(lemma))
          case _ => tokensRaw = sentence.words.toVector.
                                  map(word => word.toLowerCase).
                                  filterNot(word => this.stopWords.contains(word))
        }

        //make ngrams
        val tokens = getNGrams(tokensRaw, n, keepPunctuation = false, "_")

        //add to counter and lex
        for (t <- tokens) {

          //ensure token is lowercase
          val tLower = t.toLowerCase

          //add to lexicon
          lex.add(tLower)

          //add to counter
          if (bernoulli) {
            //only add to counter ONE time for the document
            if (!documentLex.contains(tLower)) {
              counterMap(label).incrementCount(tLower, 1)   //get an indicator count (1 or 0)
              documentCounter.incrementCount(tLower, 1)     //add to document counter
              //add to lex for this document
              documentLex.add(tLower)
            }
          } else {
            //add to counter each time token appears
            counterMap(label).incrementCount(tLower)    //get a raw frequency count
            //add to document counter only once
            if (!documentLex.contains(tLower)) {
              documentCounter.incrementCount(tLower, 1)
            }
            //add to lex for this document
            documentLex.add(tLower)
          }
        }
      }
    }

    (lex, documentCounter, counterMap)
  }


  /**
    * Get denominator for **multinomial** *conditional probabilities*
 *
    * @param c The class to be used
    * @return `P(t|c) = (count of token in class + 1) / (number of all tokens in class + size of vocabulary)` <br>
    *          `+ size of vocabulary` because the number of "bins" is equal to the size of the vocabulary
    */
  def calculateMultiCondProbDenominator(c: Int): Double = {
    this.tokenCounter(c).values.sum + this.lex.size.toDouble
  }


  /**
    * Get denominator for **bernoulli** *conditional probabilities*
    *
    * @param c The class to be used
    * @return `P(t|c) = (number of documents containing token + 1) / (total number of documents in class + 2)` <br>
    *          `+2` because there are two "bins": `occurrence` and `non-occurrence`
    */
  def calculateBernoulliCondProbDenominator(c: Int): Double = {
    this.classCounter.getCount(c) + 2
  }


  /**
    * Get *conditional probabilities* for all tokens
    */
  def getCondProbs: Map[Int, Counter[String]] = {
    (for (c <- this.tokenCounter.keySet) yield {

      //*conditional probability* denominator
      val denominator = this.condProbDenominators(c)

      //calculate *conditional probabilities*
      val condProbCounter = this.tokenCounter(c).mapValues(v => calculateCondProb(v, this.smoothingAlpha, denominator))

      //return class -> Counter
      c -> condProbCounter

    }).toMap
  }


  /**
    * Applies document feature selection to the training set <br>
    * Adjusts [[tokenCounter]] in place
    */
  def applyDocumentFrequencyFS: Unit = {

    //build new, filtered counter
    val filteredCounter =
      //for each class
      (for (c <- this.possibleClasses) yield {
      //filter tokenCounter to only include words that meet the minimum document threshold
      c -> this.tokenCounter(c).filter(token =>
        this.documentCounter.getCount(token._1) / this.docs.length.toDouble > this.documentFrequencyThreshold     //doc freq / total docs > doc freq threshold
      )
    }).toMap

    this.tokenCounter = filteredCounter
  }

}


/**
  * `Class` to test one or more documents on a trained Naive Bayes model
 *
  * @param docs The documents to be used in training
  * @param stopWords Any **lowercased** `stop words` to be filtered out of the calculations
  * @param gram `word`s or `lemma`s to be used
  * @param n Size of n-gram
  * @param bernoulli If `true`, then counts will be indicators(0 or 1); otherwise, counts will be frequency counts
  * @param smoothingAlpha if `1d` then "Laplace" smoothing, else if `<1d` then Lidstone smoothing
  * @param trainingLexicon Vocabulary from trained model; only needed if `bernoulli == true`: [[NaiveBayesTraining.lex]]
  * @param trainingClassPriors Prior class probabilities from trained model: [[NaiveBayesTraining.classPriors]]
  * @param trainingClassCounts Frequency or indicator counts for all tokens seen in training: [[NaiveBayesTraining.tokenCounter]]
  * @param trainingCondProbsDenominators Denominator used in calculating conditional probabilities: [[NaiveBayesTraining.condProbDenominators]] <br>
  *   `multinomial`: `P(t|c) = (count of token in class + 1) / (number of all tokens in class + size of vocabulary)`
  *   `bernoulli`: `P(t|c) = (number of documents containing token + 1) / (total number of documents in class + 2)`
  */
class NaiveBayesTesting(
                         val docs: Vector[Document],
                         val stopWords: Set[String] = Set(),
                         val gram: String = "word", //"word" or "lemma"
                         val n: Int = 1,
                         val bernoulli: Boolean = false,
                         val smoothingAlpha: Double = 1d, //if `1d` then Laplace, if less than `1d` then Lidstone
                         val trainingLexicon: Lexicon[String] = null,
                         val trainingClassPriors: Counter[Int],
                         val trainingClassCounts: Map[Int, Counter[String]],
                         val trainingCondProbsDenominators: Map[Int, Double]
                       ) {

  /**
    * All classes
    */
  val allClasses = this.trainingClassPriors.keySet



  /**
    * Given a counter of scores for all classes, chooses the most likely
 *
    * @param probsCounter Output from [[getProbsMulti]]
    * @return Most likely class
    */
  def predict(probsCounter: Counter[Int]): Int = {
    probsCounter.argMax._1
  }


  /**
    * Calculates the likelihood of each class for a given document in `multinomial` model
    * @param doc The [[Document]] to be analyzed
    * @param gram `word` or `lemma`
    * @param n Size of n-gram
    * @return ._1 = [[Counter]] of scores for each class <br>
    *         ._2 = [[Counter]] of raw frequency counts for each token in document
    */
  def getProbsMulti(doc: Document, gram: String, n: Int = 1): (Counter[Int], Counter[String]) = {

    //initialize new counter for predicted values for each class
    val predictedValuesCounter = new Counter[Int]()

    //initialize new counter for raw counts for each token in test document
    val rawCounts = new Counter[String]()

    //initialize counter with *log prior probabilities*
    for (c <- this.allClasses) {
      predictedValuesCounter.incrementCount(
        c,
        scala.math.log(this.trainingClassPriors.getCount(c))
      )
    }

    //initialize variable to house tokens (will be "word" or "lemma")
    var tokensRaw = Vector[String]()

    //iterate through sentences of test doc
    for (sentence <- doc.sentences) {

      //get word or lemma and filter stop words
      gram match {
        case "word" => tokensRaw = sentence.words.toVector.
                                    map(word => word.toLowerCase).
                                    filterNot(word => this.stopWords.contains(word))
        case "lemma" => tokensRaw = sentence.lemmas.get.toVector.
                                    map(lemma => lemma.toLowerCase).
                                    filterNot(lemma => this.stopWords.contains(lemma))
        case _ => tokensRaw = sentence.words.toVector.
                                map(word => word.toLowerCase).
                                filterNot(word => this.stopWords.contains(word))
      }

      //make ngrams
      val tokens = getNGrams(tokensRaw, n, keepPunctuation = false, "_")

      //iterate through tokens
      for (t <- tokens) {

        //ensure token is lowercase
        val tLower = t.toLowerCase

        //increment raw count counter
        rawCounts.incrementCount(tLower)

        //iterate through classes
        for (c <- allClasses) {

          //get denominator value for conditional probability
          val condProbDenominator = this.trainingCondProbsDenominators(c)

          //calculate *log conditional probability*
          val logCondProb = scala.math.log(calculateCondProb(this.trainingClassCounts(c).getCount(tLower), this.smoothingAlpha, condProbDenominator))

          //increment count in final counter by the *log conditional probability*
          predictedValuesCounter.incrementCount(
            c,
            logCondProb
          )

        }
      }
    }

    //return counter with accumulated scores
    (predictedValuesCounter, rawCounts)

  }


  /**
    * Calculates the likelihood of each class for a given document in `bernoulli` model
    * @param doc The [[Document]] to be analyzed
    * @param gram `word` or `lemma`
    * @param n Size of n-gram
    * @return ._1 = [[Counter]] of scores for each class <br>
    *         ._2 = [[Counter]] of indicator counts (0 or 1) for each token in document
    * @todo confirm use of `Set`s, `intersection` and `diff` is most efficient
    */
  def getProbsBernoulli(doc: Document, gram: String, n: Int = 1): (Counter[Int], Counter[String]) = {

    //initialize new counter for predicted values for each class
    val predictedValuesCounter = new Counter[Int]()

    //initialize new counter for indicator counts (0 or 1) for each token in test document
    val indicatorCounts = new Counter[String]()

    //initialize counter with *log prior probabilities*
    for (c <- this.allClasses) {
      predictedValuesCounter.incrementCount(
        c,
        scala.math.log(this.trainingClassPriors.getCount(c))
      )
    }

    //get all tokens from test document (will be "word" or "lemma")
      //ensure lowercase
    val tokens = gram match {
                    case "word" => doc.sentences.flatMap(sent => sent.words.
                                    map(word => word.toLowerCase)).
                                    filterNot(word => this.stopWords.contains(word)).
                                    toSet
                    case "lemma" => doc.sentences.flatMap(sent => sent.lemmas.get.
                                      map(_.toLowerCase)).
                                      filterNot(lemma => this.stopWords.contains(lemma)).
                                      toSet
                    case _ => doc.sentences.flatMap(sent => sent.words.
                                map(word => word.toLowerCase)).
                                filterNot(word => this.stopWords.contains(word)).
                                toSet
                  }

    //add tokens to indicatorCounter
    tokens.foreach(z => indicatorCounts.setCount(z, 1))

    //lexicon without stop words
    val lex = this.trainingLexicon.keySet
//    val lexMinusStopWords = this.trainingLexicon.keySet.diff(this.stopWords)

    //training tokens present in document
    val tokensPresent = tokens.intersect(lex)

    //training tokens not present in document
    val tokensNotPresent = lex.diff(tokens)

    //iterate through classes
    for (c <- allClasses) {

      //get denominator value for conditional probability
      val condProbDenominator = this.trainingCondProbsDenominators(c)

      //iterate through tokensPresent
      for (t <- tokensPresent) {

        //calculate *log conditional probability*
        val logCondProb = scala.math.log(calculateCondProb(this.trainingClassCounts(c).getCount(t), this.smoothingAlpha, condProbDenominator))

        //increment count in final counter by the *log conditional probability*
        predictedValuesCounter.incrementCount(
          c,
          logCondProb
        )
      }

      //iterate through tokensNotPresent
      for (t <- tokensNotPresent) {

        //calculate *log conditional probability*
        val logCondProb = scala.math.log(1 - calculateCondProb(this.trainingClassCounts(c).getCount(t), this.smoothingAlpha, condProbDenominator))

        //increment count in final counter by (1 - *log conditional probability*)
        predictedValuesCounter.incrementCount(
          c,
          logCondProb
        )
      }
    }

    //return counter with accumulated scores
    (predictedValuesCounter, indicatorCounts)

  }

}


/**
  * Support methods used in Naive Bayes
  */
object NaiveBayesSupportMethods {


  /**
    * Calculates the *conditional probability* for a raw count
    * @param count In `multinomial`, the raw frequency count of token across class; in `bernoulli` the number of documents containing token
    * @param denominator The denominator used in calculation (dependent on class)
    * @return `multinomial: P(t|c) = (count + 1) / (total size of training docs in class + total size of vocabulary)` <br>
    *         `bernoulli: P(t|c) = (count + 1) / (number of documents in class + 2)`
    */
  def calculateCondProb(count: Double, alpha: Double, denominator: Double): Double = {
    (count + alpha) / denominator
  }

}



