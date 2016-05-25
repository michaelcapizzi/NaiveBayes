package NaiveBayes

import edu.arizona.sista.processors.fastnlp.FastNLPProcessor


/**
  * `Class` for easily testing NB
  */
class testing {

  val p = new FastNLPProcessor(useMalt = true)

  val train_d_1_c = p.mkDocument("Chinese Beijing Chinese")
  val train_d_2_c = p.mkDocument("Chinese Chinese Shanghai")
  val train_d_3_c = p.mkDocument("Chinese Macao")
  val train_d_4_noC = p.mkDocument("Tokyo Japan Chinese")
  val test_d_1 = p.mkDocument("Chinese Chinese Chinese Tokyo Japan")

  val allDocs = Vector(train_d_1_c, train_d_2_c, train_d_3_c, train_d_4_noC, test_d_1)

  allDocs.foreach(p.tagPartsOfSpeech)
  allDocs.foreach(p.lemmatize)

  val nbTrainL = new NaiveBayesTraining(
    docs = allDocs.dropRight(1),
    labels = Vector(1,1,1,0),
    gram = "lemma",
    bernoulli = true,
    documentFrequencyThreshold =   .25
  )

  val nbTestL = new NaiveBayesTesting(
    docs = Vector(test_d_1),
    gram = "lemma",
    bernoulli = true,
    trainingLexicon = nbTrainL.lex,
    trainingClassPriors = nbTrainL.classPriors,
    trainingClassCounts = nbTrainL.tokenCounter,
    trainingCondProbsDenominators = nbTrainL.condProbDenominators
    )

  val nbTrainWBernoulli = new NaiveBayesTraining(
    docs = allDocs.dropRight(1),
    labels = Vector(1,1,1,0),
    bernoulli = true,
    documentFrequencyThreshold =   .25
  )

  val nbTestWBernoulli = new NaiveBayesTesting(
    docs = Vector(test_d_1),
    bernoulli = true,
    trainingLexicon = nbTrainWBernoulli.lex,
    trainingClassPriors = nbTrainWBernoulli.classPriors,
    trainingClassCounts = nbTrainWBernoulli.tokenCounter,
    trainingCondProbsDenominators = nbTrainWBernoulli.condProbDenominators
  )

  val nbTrainW = new NaiveBayesTraining(
    docs = allDocs.dropRight(1),
    labels = Vector(1,1,1,0),
    bernoulli = false,
    documentFrequencyThreshold =   .25
  )


}
