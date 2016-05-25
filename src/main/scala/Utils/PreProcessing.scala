package Utils

/**
  * Created by mcapizzi on 5/16/16.
  */
object PreProcessing {

  /**
    * Replaces any digits with "number"
    * @param text Text to be cleaned
    * @return Text with all digits replaced by "number"
    */
  def replaceNumber(text: String): String = {

    //regex to find numbers
    val numRegex = """\.*\d+[\.\/\\]*\d*""".r

    numRegex.replaceAllIn(text, "number")
  }


  /**
    * Generates `n-gram`s from a list of tokens
    * @param tokens List of tokens
    * @param n Size of the `n-gram`
    * @param keepPunctuation If `true`, punctuation is included in the `n-gram`s; else it's removed before building `n-gram`s
    * @param separator The character to split tokens in the `n-gram`
    * @return `Vector(n-grams)`
    */
  def getNGrams(tokens: Vector[String], n: Int, keepPunctuation: Boolean, separator: String): Vector[String] = {

    if (!keepPunctuation) {
        val tokensFiltered = tokens.filter(word => word.matches("[A-Za-z0-9]+"))       //remove punctuatiom

        tokensFiltered.iterator.sliding(n).toVector            //get window of n
          .map(ngram => ngram.mkString(separator))              //convert to string

    } else {

      tokens.iterator.sliding(n).toVector                    //get window of n
        .map(ngram => ngram.mkString(separator))              //convert to string
    }
  }

}
