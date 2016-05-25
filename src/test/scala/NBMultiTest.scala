import NBMultiTestValues._
import NaiveBayes.NaiveBayesSupportMethods._
import NaiveBayes.{NaiveBayesSupportMethods, NaiveBayesTesting, NaiveBayesTraining}
import SupportItems._
import edu.arizona.sista.struct.Counter

/**
  * Correct values for Naive Bayes *multinomial* classification test
  *
  * @see [http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html]
  */
object NBMultiTestValues {

  //1 = Chinese
  //0 = not Chinese

  /**
    * correct values for priors
    */
  val trainPriors = Map(
    1 -> 3d / 4d,
    0 -> 1d / 4d
  )

  /**
    * correct values for training raw frequency counts in `multinomial`
    */
  val trainRawCountsMulti = Map(
    1 -> new Counter[String](),
    0 -> new Counter[String]()
  )
  trainRawCountsMulti(1).setCount("chinese", 5)
  trainRawCountsMulti(1).setCount("beijing", 1)
  trainRawCountsMulti(1).setCount("shanghai", 1)
  trainRawCountsMulti(1).setCount("macao", 1)
  trainRawCountsMulti(0).setCount("tokyo", 1)
  trainRawCountsMulti(0).setCount("japan", 1)
  trainRawCountsMulti(0).setCount("chinese", 1)

  /**
    * correct values for testing raw frequency counts in `multinomial`
    */
  val testRawCountsMulti = new Counter[String]()
  testRawCountsMulti.setCount("chinese", 3)
  testRawCountsMulti.setCount("tokyo", 1)
  testRawCountsMulti.setCount("japan", 1)

  /**
    * correct values for conditional probability denominators in `multinomial`
    */
  val trainCondProbDenominatorsMulti = Map(
    1 -> 14d,
    0 -> 9d
  )

  /**
    * correct values for conditional probabilities in `multinomial`
    */
  val trainCondProbsMulti = Map(
    1 -> new Counter[String](),
    0 -> new Counter[String]()
  )
  trainCondProbsMulti(1).setCount("chinese", 6d / 14d)
  trainCondProbsMulti(1).setCount("beijing", 2d / 14d)
  trainCondProbsMulti(1).setCount("shanghai", 2d / 14d)
  trainCondProbsMulti(1).setCount("macao", 2d / 14d)
  trainCondProbsMulti(0).setCount("chinese", 2d / 9d)
  trainCondProbsMulti(0).setCount("tokyo", 2d / 9d)
  trainCondProbsMulti(0).setCount("japan", 2d / 9d)

  /**
    * correct values for log conditional probabilities in `multinomial`<br>
    * as calculated by [[scala.math.log]]
    */
  val trainLogCondProbsMulti = Map(
    1 -> new Counter[String](),
    0 -> new Counter[String]()
  )
  trainLogCondProbsMulti(1).setCount("chinese", scala.math.log(trainCondProbsMulti(1).getCount("chinese")))
  trainLogCondProbsMulti(1).setCount("beijing", scala.math.log(trainCondProbsMulti(1).getCount("beijing")))
  trainLogCondProbsMulti(1).setCount("shanghai", scala.math.log(trainCondProbsMulti(1).getCount("shanghai")))
  trainLogCondProbsMulti(1).setCount("macao", scala.math.log(trainCondProbsMulti(1).getCount("macao")))
  trainLogCondProbsMulti(0).setCount("chinese", scala.math.log(trainCondProbsMulti(0).getCount("chinese")))
  trainLogCondProbsMulti(0).setCount("tokyo", scala.math.log(trainCondProbsMulti(0).getCount("tokyo")))
  trainLogCondProbsMulti(0).setCount("japan", scala.math.log(trainCondProbsMulti(0).getCount("japan")))


  /**
    * Final value for class 'Chinese' on test document in `multinomial` = -8.1076
    */
  val cMulti = scala.math.log(trainPriors(1)) + //log prior = -0.2877
    scala.math.log(6d / 14d) * 3 + //log cond prob for "Chinese" = -.8473 * 3 = -2.5419
    scala.math.log(1d / 14d) + //log cond prob for "Tokyo" = -2.6390
    scala.math.log(1d / 14d) //log cond prob for "Japan" = -2.6390


  /**
    * Final value for class 'not Chinese' on test document in `multinomial`= -8.9067
    */
  val noCMulti = scala.math.log(trainPriors(0)) + //log prior= -1.3863
    scala.math.log(2d / 9d) * 3 + //log cond prob for "Chinese" = -1.5041 * 3 = -4.5122
    scala.math.log(2d / 9d) + //log cond prob for "Tokyo" = -1.5041
    scala.math.log(2d / 9d) //log cond prob for "Japan" = -1.5041

}


/**
  * Suite of tests to confirm the calculations made by [[NaiveBayesTraining]] and [[NaiveBayesTesting]] for *multinomial*
  */
class NBMultiTest extends GeneralTest {

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


      /**
        * [[NaiveBayesTraining]] with use of `word` and no stop words and `multinomial`
        */
      val nbTrainNoStopWordMulti = new NaiveBayesTraining(
        docs = Vector(train_d_1_c, train_d_2_c, train_d_3_c, train_d_4_noC),
        labels = Vector(1,1,1,0)
      )


      /**
        * [[NaiveBayesTesting]] with use of `word` and no stop words and `multinomial`
        */
      val nbTestNoStopWordMulti = new NaiveBayesTesting(
        docs = Vector(test_d_1),
        trainingClassPriors = nbTrainNoStopWordMulti.classPriors,
        trainingClassCounts = nbTrainNoStopWordMulti.tokenCounter,
        trainingCondProbsDenominators = nbTrainNoStopWordMulti.condProbDenominators
      )


    }


  ////////////////////////////////////////////////////////////////////////////////////




  "Conditional probabilities" should "be correctly calculated by support method" in {
    assert(calculateCondProb(5d, 1d, 14d) == trainCondProbsMulti(1).getCount("chinese"))
  }


  "Log conditional probabilities" should "be correctly calculated by support method" in {
    assert(scala.math.log(calculateCondProb(5d, 1d, 14d)) == trainLogCondProbsMulti(1).getCount("chinese"))
  }


  "Number of documents" should "be 4" in {assert(required.nbTrainNoStopWordMulti.numberOfDocs == 4)
  }


  "Training class Priors" should "be correctly calculated" in {
    assert(
      required.nbTrainNoStopWordMulti.classPriors.getCount(1) == trainPriors(1) &&
      required.nbTrainNoStopWordMulti.classPriors.getCount(0) == trainPriors(0)
    )
  }


  "Training class raw frequency counts" should "be correctly calculated" in {
    assert(required.nbTrainNoStopWordMulti.tokenCounter == trainRawCountsMulti)
  }


  "Training *multinomial* class conditional probability denominators" should "be correctly calculated" in {
    assert(required.nbTrainNoStopWordMulti.condProbDenominators == trainCondProbDenominatorsMulti)
  }


  "Training *multinomial* conditional probabilities" should "be correctly calculated" in {
    assert(required.nbTrainNoStopWordMulti.condProbCounters == trainCondProbsMulti)
  }



  "Testing raw frequency counts" should "be correctly calculated" in {
    val chinese = 3d
    val tokyo = 1d
    val japan = 1d

    val rawCounts = required.nbTestNoStopWordMulti.getProbsMulti(required.test_d_1, required.nbTestNoStopWordMulti.gram)._2

    assert(
            rawCounts.getCount("chinese") == 3d &&
            rawCounts.getCount("tokyo") == 1d &&
            rawCounts.getCount("japan") == 1d
          )
  }


  "Naive Bayes" should "generate correct value for *multinomial* likelihood of being class 'Chinese' on the test document" in {
    val chineseLikelihood = required.nbTestNoStopWordMulti.getProbsMulti(required.test_d_1, required.nbTestNoStopWordMulti.gram)._1.getCount(1)

    assert(chineseLikelihood == cMulti)
  }


  "Naive Bayes" should "generate correct value for *multinomial* likelihood of being class 'not Chinese' on the test document" in {
    val noChineseLikelihood = required.nbTestNoStopWordMulti.getProbsMulti(required.test_d_1, required.nbTestNoStopWordMulti.gram)._1.getCount(0)

    assert(noChineseLikelihood == noCMulti)
  }


  "Naive Bayes" should "correctly predict the *multinomial* class `Chinese` for the test document" in {

    val probs = required.nbTestNoStopWordMulti.getProbsMulti(required.test_d_1, required.nbTestNoStopWordMulti.gram)._1

    assert(required.nbTestNoStopWordMulti.predict(probs) == 1)
  }









}
