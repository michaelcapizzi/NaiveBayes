import edu.arizona.sista.processors.fastnlp.FastNLPProcessor
import org.scalatest._

/**
  * Support items for all tests
  */
object SupportItems {

  /**
    * [[FastNLPProcessor]]
    */
  val p = new FastNLPProcessor(useMalt = true)

}


/**
  * Abstract class for all tests
  */
abstract class GeneralTest extends FlatSpec with Matchers with OptionValues with Inside with Inspectors {

}
