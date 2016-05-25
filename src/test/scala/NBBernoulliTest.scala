import NBBernoullitTestValues._
import NaiveBayes.{NaiveBayesTesting, NaiveBayesTraining}
import SupportItems._
import edu.arizona.sista.struct.Counter

/**
  * Correct values for Naive Bayes *multinomial* classification test
  *
  * @see [http://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html]
  */
object NBBernoullitTestValues {

  //1 = Chinese
  //0 = not Chinese

  /**
    * correct values for priors
    */
  val trainPriors = Map(
    1 -> 3d/4d,
    0 -> 1d/4d
  )

  /**
    * correct values for training raw indicator counts in `bernoulli`
    */
  val trainCountsBernoulli = Map(
    1 -> new Counter[String](),
    0 -> new Counter[String]()
  )
  trainCountsBernoulli(1).setCount("chinese", 3)
  trainCountsBernoulli(1).setCount("beijing", 1)
  trainCountsBernoulli(1).setCount("shanghai", 1)
  trainCountsBernoulli(1).setCount("macao", 1)
  trainCountsBernoulli(0).setCount("tokyo", 1)
  trainCountsBernoulli(0).setCount("japan", 1)
  trainCountsBernoulli(0).setCount("chinese", 1)

  /**
    * correct values for testing raw indicator counts in `bernoulli`
    */
  val testCountsBernoulli = new Counter[String]()
  testCountsBernoulli.setCount("chinese", 1)
  testCountsBernoulli.setCount("tokyo", 1)
  testCountsBernoulli.setCount("japan", 1)

  /**
    * correct values for conditional probability denominators in `bernoulli`
    */
  val trainCondProbDenominatorsBernoulli = Map(
    1 -> 5d,
    0 -> 3d
  )

  /**
    * correct values for conditional probabilities in `bernoulli`
    */
  val trainCondProbsBernoulli = Map(
    1 -> new Counter[String](),
    0 -> new Counter[String]()
  )
  trainCondProbsBernoulli(1).setCount("chinese", 4d/5d)
  trainCondProbsBernoulli(1).setCount("beijing", 2d/5d)
  trainCondProbsBernoulli(1).setCount("shanghai", 2d/5d)
  trainCondProbsBernoulli(1).setCount("macao", 2d/5d)
  trainCondProbsBernoulli(0).setCount("chinese", 2d/3d)
  trainCondProbsBernoulli(0).setCount("tokyo", 2d/3d)
  trainCondProbsBernoulli(0).setCount("japan", 2d/3d)


  /**
    * Final value for class `Chinese` on test document in `bernoulli` = -5.262
    */
  val cBernoulli =  scala.math.log(trainPriors(1)) +        //log prior = -0.2877
    scala.math.log(4d/5d) +                 //log cond prob of "Chinese" = -.2231
    scala.math.log(1d/5d) +                 //log cond prob of "Japan" = -1.6094
    scala.math.log(1d/5d) +                 //log cond prob of "Tokyo" = -1.6094
    scala.math.log(1 - 2d/5d) +             //log (1 - cond prob of "Beijing") = -.5108
    scala.math.log(1 - 2d/5d) +             //log (1  - cond prob of "Shanghai") = -.5108
    scala.math.log(1 - 2d/5d)               //log (1 - cond prob of "Macao") = -.5108


  /**
    * Final value for class `Chinese` on test document in `bernoulli` = -3.8193
    */
  val noCBernoulli =  scala.math.log(trainPriors(0)) +        //log prior= -1.3863
    scala.math.log(2d/3d) +                 //log cond prob of "Chinese" = -.4055
    scala.math.log(2d/3d) +                 //log cond prob of "Japan" = -.4055
    scala.math.log(2d/3d) +                 //log cond prob of "Tokyo" = -.4055
    scala.math.log(1 - 1d/3d) +           //log (1 - cond prob of "Beijing") = -.4055
    scala.math.log(1 - 1d/3d) +           //log (1 - cond prob of "Shanghai") = -.4055
    scala.math.log(1 - 1d/3d)             //log (1 - cond prob of "Macao") = -.4055


}

/**
  * Suite of tests to confirm the calculations made by [[NaiveBayesTraining]] and [[NaiveBayesTesting]] for *bernoulli*
  */
class NBBernoulliTest extends GeneralTest {

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
        * [[NaiveBayesTraining]] with use of `word` and no stop words and `bernoulli`
        */
      val nbTrainNoStopWordBernoulli = new NaiveBayesTraining(
        docs = Vector(train_d_1_c, train_d_2_c, train_d_3_c, train_d_4_noC),
        labels = Vector(1, 1, 1, 0),
        bernoulli = true
      )


      /**
        * [[NaiveBayesTesting]] with use of `word` and no stop words and `bernoulli`
        */
      val nbTestNoStopWordBernoulli = new NaiveBayesTesting(
        docs = Vector(test_d_1),
        bernoulli = true,
        trainingLexicon = nbTrainNoStopWordBernoulli.lex,
        trainingClassPriors = nbTrainNoStopWordBernoulli.classPriors,
        trainingClassCounts = nbTrainNoStopWordBernoulli.tokenCounter,
        trainingCondProbsDenominators = nbTrainNoStopWordBernoulli.condProbDenominators
      )

    }

  ////////////////////////////////////////////////////////////////////////////////////

  "Training *bernoulli* class conditional probability denominators" should "be correctly calculated" in {
    assert(required.nbTrainNoStopWordBernoulli.condProbDenominators == trainCondProbDenominatorsBernoulli)
  }

  "Training *bernoulli* conditional probabilities" should "be correctly calculated" in {
    assert(required.nbTrainNoStopWordBernoulli.condProbCounters == trainCondProbsBernoulli)
  }

  "Naive Bayes" should "generate correct value for *bernoulli* likelihood of being class 'Chinese' on the test document" in {
    val chineseLikelihood = required.nbTestNoStopWordBernoulli.getProbsBernoulli(required.test_d_1, required.nbTestNoStopWordBernoulli.gram)._1.getCount(1)

    assert(chineseLikelihood == cBernoulli)
  }

  "Naive Bayes" should "generate correct value for *bernoulli* likelihood of being class 'not Chinese' on the test document" in {
    val chineseLikelihood = required.nbTestNoStopWordBernoulli.getProbsBernoulli(required.test_d_1, required.nbTestNoStopWordBernoulli.gram)._1.getCount(0)

    assert(chineseLikelihood == noCBernoulli)
  }


  "Naive Bayes" should "correctly predict the *bernoulli* class `not Chinese` for the test document" in {

    val probs = required.nbTestNoStopWordBernoulli.getProbsBernoulli(required.test_d_1, required.nbTestNoStopWordBernoulli.gram)._1

    assert(required.nbTestNoStopWordBernoulli.predict(probs) == 0)
  }



}
