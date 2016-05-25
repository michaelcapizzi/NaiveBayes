import NaiveBayes.{NaiveBayesTesting, NaiveBayesTraining}
import SupportItems._

class NBMiscTest extends GeneralTest {

  def required =
    new {

      val train_d_1_c = p.mkDocument(scala.io.Source.fromInputStream(getClass.getResourceAsStream("/NB_documents/train/train_1_c.txt")).mkString)
      val train_d_2_c = p.mkDocument(scala.io.Source.fromInputStream(getClass.getResourceAsStream("/NB_documents/train/train_2_c.txt")).mkString)
      val train_d_3_c = p.mkDocument(scala.io.Source.fromInputStream(getClass.getResourceAsStream("/NB_documents/train/train_3_c.txt")).mkString)
      val train_d_4_noC = p.mkDocument(scala.io.Source.fromInputStream(getClass.getResourceAsStream("/NB_documents/train/train_4_noC.txt")).mkString)
      val test_d_1 = p.mkDocument(scala.io.Source.fromInputStream(getClass.getResourceAsStream("/NB_documents/test/test_1.txt")).mkString)

      val allDocs = Vector(train_d_1_c, train_d_2_c, train_d_3_c, train_d_4_noC, test_d_1)

      allDocs.foreach(p.tagPartsOfSpeech)
      allDocs.foreach(p.lemmatize)

      val lemmaTest = p.mkDocument("I am testing lemmas here")

      p.tagPartsOfSpeech(lemmaTest)
      p.lemmatize(lemmaTest)


      /**
        * [[NaiveBayesTraining]] with use of `word` and no stop words and `multinomial`
        */
      val nbTrainNoStopWordMulti = new NaiveBayesTraining(
        docs = Vector(train_d_1_c, train_d_2_c, train_d_3_c, train_d_4_noC),
        labels = Vector(1,1,1,0)
      )


      /**
        * [[NaiveBayesTraining]] with use of `lemma` and no stop words <br>
        * Only used for testing `lemma` capacity <br>
        * Should *not* be used in testing full NB
        */
      val nbTrainNoStopLemma = new NaiveBayesTraining(
        docs = Vector(lemmaTest),
        labels = Vector(1),
        gram = "lemma"
      )


      /**
        * [[NaiveBayesTraining]] with use of `word` and filtering stop words <br>
        * Only used for testing stop word filtering capacity <br>
        * Should *not* be used in testing full NB
        */
      val nbTrainStopWord = new NaiveBayesTraining(
        docs = Vector(train_d_1_c, train_d_2_c, train_d_3_c, train_d_4_noC),
        labels = Vector(1,1,1,0),
        stopWords = Set("macao")
      )


      /**
        * [[NaiveBayesTesting]] with use of `word` and filtering stop words <br>
        * Only used for testing stop word filtering capacity <br>
        * Should *not* be used in testing full NB
        */
      val nbTestStopWord = new NaiveBayesTesting(
        docs = Vector(test_d_1),
        trainingClassPriors = nbTrainNoStopWordMulti.classPriors,
        trainingClassCounts = nbTrainNoStopWordMulti.tokenCounter,
        trainingCondProbsDenominators = nbTrainNoStopWordMulti.condProbDenominators,
        stopWords = Set("tokyo")
      )


      /**
        * [[NaiveBayesTraining]] for testing *multinomial* `documentFrequency` feature selection
        */
//      val nbTrainMultiDF = new NaiveBayesTraining(
//        docs = Vector(train_d_1_c, train_d_2_c, train_d_3_c, train_d_4_noC),
//        labels = Vector(1,1,1,0)
//      )


      /**
        * [[NaiveBayesTesting]] for testing *multinomial* `documentFrequency` feature selection
        */
//      val nbTestMultiDF = new NaiveBayesTesting(
//        docs = Vector(test_d_1),
//        trainingClassPriors = nbTrainMultiDF.classPriors,
//        trainingClassCounts = nbTrainMultiDF.tokenCounter,
//        trainingCondProbsDenominators = nbTrainMultiDF.condProbDenominators
//      )


//      /**
//        * [[NaiveBayesTraining]] for testing *bernoulli* `documentFrequency` feature selection
//        */
//      val nbTrainBernoulliDF = new NaiveBayesTraining(
//        docs = Vector(train_d_1_c, train_d_2_c, train_d_3_c, train_d_4_noC),
//        labels = Vector(1,1,1,0),
//        bernoulli = true,
//        documentFrequencyThreshold = .25
//      )


      /**
        * [[NaiveBayesTesting]] for testing *bernoulli* `documentFrequency` feature selection
        */
//      val nbTestBernoulliDF = new NaiveBayesTesting(
//        docs = Vector(test_d_1),
//        bernoulli = true,
//        trainingLexicon = nbTrainBernoulliDF.lex,
//        trainingClassPriors = nbTrainBernoulliDF.classPriors,
//        trainingClassCounts = nbTrainBernoulliDF.tokenCounter,
//        trainingCondProbsDenominators = nbTrainBernoulliDF.condProbDenominators
//      )

    }


  ////////////////////////////////////////////////////////////////////////////////////


  "Naive Bayes algorithm" should "be able to handle lemmas" in {
    val lemmas = required.nbTrainNoStopLemma.tokenCounter(1).keySet

    assert(!lemmas.contains("am") && lemmas.contains("be"))
  }


  "Naive Bayes algorithm" should "properly filter stop words" in {
    val macaoTrainingCount = required.nbTrainStopWord.tokenCounter(1).getCount("macao")
    val tokyoTestingCount = required.nbTestStopWord.getProbsMulti(required.nbTestStopWord.docs.head, required.nbTestStopWord.gram)._2.getCount("tokyo")

    assert(macaoTrainingCount == 0 && tokyoTestingCount == 0)
  }


  //In order to make adjustments to tokenCounter in place, the class has to appear inside this test
  "Document Frequency feature selection for *bernoulli*" should "reduce the size of the training vocabulary" in {

    val nbTrainBernoulliDF = new NaiveBayesTraining(
      docs = Vector(required.train_d_1_c, required.train_d_2_c, required.train_d_3_c, required.train_d_4_noC),
      labels = Vector(1,1,1,0),
      bernoulli = true,
      documentFrequencyThreshold = .25
    )

    val counterBeforeDF = nbTrainBernoulliDF.tokenCounter

    //apply DF feature selection
    nbTrainBernoulliDF.applyDocumentFrequencyFS

    val counterAfterDF = nbTrainBernoulliDF.tokenCounter

    assert(
      counterAfterDF(1).size < counterBeforeDF(1).size &&
      counterAfterDF(0).size < counterBeforeDF(0).size &&
      counterAfterDF(1).keySet == Set("chinese") &&
      counterAfterDF(0).keySet == Set("chinese")
    )

  }


  "Document Frequency feature selection for *multinomial*" should "reduce the size of the training vocabulary" in {

    val nbTrainMultiDF = new NaiveBayesTraining(
      docs = Vector(required.train_d_1_c, required.train_d_2_c, required.train_d_3_c, required.train_d_4_noC),
      labels = Vector(1,1,1,0),
      documentFrequencyThreshold = .25
    )

    val counterBeforeDFMulti = nbTrainMultiDF.tokenCounter

    //apply DF feature selection
    nbTrainMultiDF.applyDocumentFrequencyFS

    val counterAfterDFMulti = nbTrainMultiDF.tokenCounter

    assert(
      counterAfterDFMulti(1).size < counterBeforeDFMulti(1).size &&
      counterAfterDFMulti(0).size < counterBeforeDFMulti(0).size &&
      counterAfterDFMulti(1).keySet == Set("chinese") &&
      counterAfterDFMulti(0).keySet == Set("chinese")
    )

  }


  "Lidstone smoothing" should "be reduce the conditional probability in comparison to Laplace smoothing" in {
    val nbLaplace = new NaiveBayesTraining(
      docs = Vector(required.train_d_1_c, required.train_d_2_c, required.train_d_3_c, required.train_d_4_noC),
      labels = Vector(1,1,1,0),
      smoothingAlpha = 1d
    )

    val nbLidstone = new NaiveBayesTraining(
      docs = Vector(required.train_d_1_c, required.train_d_2_c, required.train_d_3_c, required.train_d_4_noC),
      labels = Vector(1,1,1,0),
      smoothingAlpha =   .5
    )

    assert(nbLaplace.condProbCounters(0).getCount("chinese") > nbLidstone.condProbCounters(0).getCount("chinese"))

  }

}
