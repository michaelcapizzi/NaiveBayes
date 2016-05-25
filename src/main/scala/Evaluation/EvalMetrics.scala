package Evaluation

/**
  * Contains methods for evaluation in terms of `precision`, `recall`, and `f1`
  * @param mlScoreList `Vector` of `Tuples` of `(title, predicted label, gold label)`
  */

class EvalMetrics(
                   mlScoreList: Vector[(String, Int, Int)]     //(title, mlScore, actualScore)
                 ) {

  /**
    * Makes a list of the possible labels
    */
  val possibleLabels = this.mlScoreList.map(_._3).distinct


  /**
    * Measure of accuracy (correct / total)
    * @return Percent correct
    */
  def accuracy: Double = {
    //helper function for determining correct score
    def isAccurate(mlScore: Int, actualScore: Int): Int = {
      if (mlScore == actualScore) 1 else 0
    }


    (
      this.mlScoreList.map(item =>
        isAccurate(item._2, item._3)
      ).sum.toDouble /                                //sum of all correct items
        this.mlScoreList.length.toDouble) * 100          //divided by total number of items then multiplied by 100
  }


  /**
    * Generates a label of `true positive`, `true negative`, `false positive`, and `false positive`
    */
  def relevanceLabels: Map[Int, Vector[String]] = {
    //helper function to determine label
    def determineRelevanceLabels(relevantClass: Int, mlScore: Int, actualScore: Int): String = {
      if (relevantClass == actualScore & relevantClass == mlScore) "truePositive"           //it was relevant, and it was correctly scored as relevant
      else if (relevantClass != actualScore & relevantClass == mlScore) "falsePositive"     //it was irrelevant, but it was incorrectly scored as relevant
      else if (relevantClass == actualScore & relevantClass != mlScore) "falseNegative"     //it was relevant, but it was incorrectly scored as irrelevant
      else "trueNegative"
    }

    //iterate through data points
    (for (label <- this.possibleLabels) yield {
      label -> this.mlScoreList.map(score => determineRelevanceLabels(label, score._2, score._3))      //generate relevance tags for each item
    }).toMap                                                                                        //convert to a Map
  }


  /**
    * Recall - how many actual instances were predicted correctly
    * @return `Map` of `(type, recall score)`
    */
  def recall: Map[Int, Double] = {
    //helper function for calculating recall
    def calculateRecall(truePositive:Double, falseNegative: Double): Double = {
      if ((truePositive + falseNegative) == 0) 0                                            //in case denominator is 0
      else truePositive / (truePositive + falseNegative)                                    //otherwise calculate recall
    }

    //get relevance label map
    val relevanceLabelsMap = relevanceLabels

    //iterate through data points
    (for (relevance <- relevanceLabelsMap.keySet.toList) yield {
      (
        relevance,
        calculateRecall(
          relevanceLabelsMap(relevance).count(_.matches("truePositive")).toDouble,
          relevanceLabelsMap(relevance).count(_.matches("falseNegative")).toDouble
        )
        )
    }).toMap      //convert to Map
  }


  /**
    * Precision - how many predicted instances were correct
    * @return `Map` of `(type, precision score)`
    */
  def precision: Map[Int, Double] = {
    //helper function for calculating precision
    def calculatePrecision(truePositive: Double, falsePositive: Double): Double = {
      if ((truePositive + falsePositive) == 0) 0 //in case denominator is 0
      else truePositive / (truePositive + falsePositive) //otherwise calculate recall
    }

    //get relevance label map
    val relevanceLabelsMap = relevanceLabels

    //iterate through data points
    (for (relevance <- relevanceLabelsMap.keySet.toList) yield {
      (
        relevance,
        calculatePrecision(
          relevanceLabelsMap(relevance).count(_.matches("truePositive")).toDouble,
          relevanceLabelsMap(relevance).count(_.matches("falsePositive")).toDouble
        )
        )
    }).toMap    //convert to Map
  }

  /**
    * F1 = (2 * precision * recall) / (precision + recall)
    * @param precisionScore As calculated by [[precision]]
    * @param recallScore As calculated by [[recall]]
    */
  def calculateF1(precisionScore: Double, recallScore: Double): Double = {
    if ((precisionScore + recallScore) == 0) 0                                            //in case denominator is 0
    else (2 * precisionScore * recallScore) / (precisionScore + recallScore)              //otherwise calculate recall
  }


  /**
    * Generates F1 score for each type
    * @return `Map` of `(type, f1 score)`
    */
  def f1: Map[Int, Double] = {
    val relevanceLabelsMap = relevanceLabels
    (for (relevance <- relevanceLabelsMap.keySet.toList) yield {
      val precisionScore = precision(relevance)
      val recallScore = recall(relevance)
      relevance -> calculateF1(precisionScore, recallScore)
    }).toMap
  }


  /**
    * Calculates macro scores for `precision`, `recall`, and `f1` <br>
    *   Calculated as the average of score for each type
    * @return `Map` of `(type, f1 score)`
    */
  def macroScores: Map[String, Double] = {
    val macroPrecision = precision.values.toList.sum / possibleLabels.length
    val macroRecall = recall.values.toList.sum / possibleLabels.length
    Map(
      "macroPrecision" -> macroPrecision,
      "macroRecall" -> macroRecall,
      "macroF1" -> calculateF1(macroPrecision, macroRecall)
    )
  }

  //NOTE: the description below may be inaccurate...

  //to extract
  //metrics(key) --> will yield you an Any of all three
  //to go deeper
  //metrics(key).asInstanceOf[Map[???]](key)

}
