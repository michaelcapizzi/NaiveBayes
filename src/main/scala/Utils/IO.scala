package Utils

import Utils.PreProcessing._

/**
  * Methods for reading in and writing out
  */
object IO {

  /**
    * import a `.txt` file
    *
    * @param fileName Full path to `.txt` file
    * @return The full text
    */
  def importText(fileName: String): String = {
    scala.io.Source.fromFile(fileName).getLines.mkString("\n")
  }


  /**
    * import a `.csv` file <br>
    *   First line is header <br>
    *   Each subsequent line has only *two* columns: `text`, `label`
    * @param fileName Full path to `.csv` file
    * @param delimiter Delimiter used to separate columns
    * @param split Percentage of data to split into `Vector(train, dev, test)`
    * @return The dataset in `Tuple(Tuple(text, int-label), Tuple(text, int-label), Tuple(text, int-label))` <br>
    *           ._1 = train <br>
    *           ._2 = dev <br>
    *           ._3 = test
    */
  def importCSV(fileName: String, delimiter: String, split: Vector[Double]): (Vector[(String, Int)], Vector[(String, Int)], Vector[(String, Int)]) = {
    val train = (split(0) * 10).toInt
    val dev = (split(1) * 10).toInt
    val test = (split(2) * 10).toInt

    //build ranges
    val trainR = Range(0, train)
    val devR = Range(train, train + dev)
    val testR = Range(dev + train, 10)

    //buffers to house `(text, int-label)`
    val trainBuffer = collection.mutable.Buffer[(String, Int)]()
    val devBuffer = collection.mutable.Buffer[(String, Int)]()
    val testBuffer = collection.mutable.Buffer[(String, Int)]()

    val in = scala.io.Source.fromFile(fileName).getLines

    var c = 0

    for (line <- in.drop(1)) {
      val split = line.split(delimiter)
      val text = replaceNumber(split.head.trim())
      val label = split.last.toInt
      if (trainR.contains(c)) {
        trainBuffer += ((text, label))
      } else if (devR.contains(c)) {
        devBuffer += ((text, label))
      } else {
        testBuffer += ((text, label))
      }
      //increment or reset counter
      if (c == 9) c = 0 else c += 1
    }

    (trainBuffer.toVector, devBuffer.toVector, testBuffer.toVector)

  }


  /**
    * Imports a word list from `resources/`
    * @param resourceName Name of word list in `resources/`
    * @param comment If not `None`, then the symbol indicates a comment line to be ignored
    * @return List of words
    */
  def importWordList(resourceName: String, comment: Option[String]): Vector[String] = {

    val raw = scala.io.Source.fromInputStream(getClass.getResourceAsStream("/wordLists/" + resourceName)).getLines.toVector

    if (comment.isDefined) {
      raw.filterNot(_.startsWith(comment.get))
    } else {
      raw
    }
  }





}
