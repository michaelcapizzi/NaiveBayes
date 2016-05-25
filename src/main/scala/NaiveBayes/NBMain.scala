package NaiveBayes

import Utils.IO._
import Evaluation._
import edu.arizona.sista.processors.fastnlp.FastNLPProcessor


/**
  * Created by mcapizzi on 5/16/16.
  */
object NBMain {

  /**
    * @param args args(0) = full path to `.csv`
    *             args(1) = delimiter
    *             args(2) = splits for train/dev/test
    *               {{{
    *                 .8/.1/.1
    *               }}}
    *             args(3) = filter stop words (Boolean)
    *             args(4) = "word" or "lemma"
    *             args(5) = size of n-gram
    *             args(6) = "bernoulli" or "multi"
    *             args(7) = smoothing alpha
    *             args(8) = document frequency threshold (Percentage)
    */
  def main(args: Array[String]) = {

    val start = System.currentTimeMillis().toDouble

    //import dataset

    val (train, dev, test) = importCSV(args(0), args(1), args(2).split("/").map(_.toDouble).toVector)

    println("size of training data: " + train.length)
    println("percent of IS_MARKET: " + (train.count(_._2 == 1).toDouble / train.length.toDouble).toString.take(4))
    println("size of dev data: " + dev.length)
    println("percent of IS_MARKET: " + (dev.count(_._2 == 1).toDouble / dev.length.toDouble).toString.take(4))
    println("size of test data: " + test.length)
    println("percent of IS_MARKET: " + (test.count(_._2 == 1).toDouble / test.length.toDouble).toString.take(4))

    //////////////////////////////////////////////////////////

    //hyperparameters

    val stopWords = if (args(3).toBoolean) {
                        importWordList("stopWords.txt", Some("#")).toSet
                    } else {
                        Set[String]()
                    }

    println("Stop words used: ")
    println(stopWords)

    val gram = args(4)

    val n = args(5).toInt

    val useBernoulli = if (args(6).contains("multi")) false else true

    val alpha = args(7).toDouble

    val dfFS = args(8).toDouble

    //////////////////////////////////////////////////////////

    //process documents

    val p = new FastNLPProcessor(useMalt = true)

    val docStart = System.currentTimeMillis().toDouble

    println("preparing documents")
    val trainLabels = train.map(_._2)
    val trainDocs = train.map(_._1).map(p.mkDocument(_))

    val devLabels = dev.map(_._2)
    val devDocs = dev.map(_._1).map(p.mkDocument(_))

    val testLabels = test.map(_._2)
    val testDocs = test.map(_._1).map(p.mkDocument(_))

    if (gram == "lemma") {
      println("lemmatizing documents")
      trainDocs.foreach(p.tagPartsOfSpeech)
      trainDocs.foreach(p.lemmatize)
      devDocs.foreach(p.tagPartsOfSpeech)
      devDocs.foreach(p.lemmatize)
      testDocs.foreach(p.tagPartsOfSpeech)
      testDocs.foreach(p.lemmatize)
    }

    val docStop = System.currentTimeMillis().toDouble
    val docElapsed = (docStop - docStart) / 1000d


    //////////////////////////////////////////////////////////

    //train NB

    val nbStart = System.currentTimeMillis().toDouble

    val nbTrain = new NaiveBayesTraining(
      docs = trainDocs,
      labels = trainLabels,
      stopWords = stopWords,
      gram = gram,
      n = n,
      bernoulli = useBernoulli,
      smoothingAlpha = alpha,
      documentFrequencyThreshold = dfFS
    )

    //apply document frequency feature selection if required
    if (nbTrain.documentFrequencyThreshold > 0d) {
      println("applying feature selection: document frequency")
      nbTrain.applyDocumentFrequencyFS
    }

    val nbStop = System.currentTimeMillis().toDouble
    val trainElapsed = (nbStop - nbStart) / 1000d


    //////////////////////////////////////////////////////////

    //dev NB

    val devStart = System.currentTimeMillis().toDouble

    val nbDev = new NaiveBayesTesting(
      docs = devDocs,
      stopWords = stopWords,
      gram = gram,
      n = n,
      bernoulli = useBernoulli,
      smoothingAlpha = alpha,
      trainingLexicon = if (useBernoulli) nbTrain.lex else null,
      trainingClassPriors = nbTrain.classPriors,
      trainingClassCounts = nbTrain.tokenCounter,
      trainingCondProbsDenominators = nbTrain.condProbDenominators
    )

    val allCondProbs = if (useBernoulli) {
      devDocs.map(each => nbDev.getProbsBernoulli(each, gram, n)._1)
    } else {
      devDocs.map(each => nbDev.getProbsMulti(each, gram, n)._1)
    }

    val predictions = allCondProbs.map(each => nbDev.predict(each))

    val devStop = System.currentTimeMillis().toDouble
    val devElapsed = (devStop - devStart) / 1000d


    //////////////////////////////////////////////////////////

    val stop = System.currentTimeMillis().toDouble
    val totalElapsed = (stop - start) / 1000d

    val results = (
                    dev.map(_._1),        //text
                    predictions,          //predicted label
                    devLabels             //actual label
                  ).zipped.toVector

    val eval = new EvalMetrics(results)

    //reporting
    println("Stop words used: ")
    println(stopWords)
    println("using: " + gram + " as " + n + "-grams")
    println("is Bernoulli? " + useBernoulli.toString)
    println("smoothing alpha: " + alpha)
    println("token must appear in at least: " + (dfFS * train.length).toInt + " documents")
    println()

    println("document processing time: " + docElapsed.toString.take(4))
    println("Naive Bayes training time: " + trainElapsed.toString.take(4))
    println("total time in seconds: " + totalElapsed.toString.take(4))
    println()

    println("accuracy: " + eval.accuracy)
    println("precision: " + eval.precision(1))
    println("recall: " + eval.recall(1))
    println("f1: " + eval.f1(1))
    println()

    val falseNegatives = results.filter(z => z._2 == 0 && z._3 == 1)
    println("false negatives: " + falseNegatives.length)
    println(falseNegatives.length.toDouble / dev.count(_._2 == 1) + " percent of MARKET_RELATED")
    falseNegatives.foreach(println)
    println()

    val falsePositives =  results.filter(z => z._2 == 1 && z._3 == 0)
    println("false positives: " + falsePositives.length)
    println(falsePositives.length.toDouble / dev.count(_._2 == 0) + " percent of NOT MARKET_RELATED")
    falsePositives.foreach(println)

    println("top 50 tokens in the MARKET_RELATED class")
    println(nbTrain.condProbCounters(1).topKeys(100))
  }
}
