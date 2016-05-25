package Utils

import java.io.ByteArrayInputStream
import java.util.Properties
import javax.mail.Session
import javax.mail.internet.MimeMessage


/**
  * `Class` to implement `javax.mail`
  * @see [http://stackoverflow.com/questions/3444660/java-email-message-parser]
  * @see [https://java.net/projects/javamail/pages/Home#Project_Documentation]
  */
class ParseEmail {

  /**
    * Session to start the process
    */
  val session = Session.getDefaultInstance(new Properties())


  /**
    * Convert raw message to [[ByteArrayInputStream]]
    * @param content Raw message
    */
  def getBytes(content: String): ByteArrayInputStream = {
    new ByteArrayInputStream(content.getBytes())
  }


  /**
    * Convert [[ByteArrayInputStream]] to [[javax.mail.Message]]
    * @param inputStream Output of [[getBytes]]
    */
  def getMessage(inputStream: ByteArrayInputStream): MimeMessage = {
    new MimeMessage(this.session, inputStream)
  }


  /**
    * Get just the message text <br>
    * **Note**: This *cannot* distinguish between forwarded/replied in-line messages from new message
    * @param message Output of [[getMessage]]
    */
  def getMessageText(message: MimeMessage): String = {
    message.getContent.toString()
  }

  ///////////////////////////////

  val emailNoReply = scala.io.Source.fromFile("/media/mcapizzi/data/Google_Drive_Arizona/Corpora/enron_full/allen-p/inbox/1.").getLines.mkString("\n")

  val m = getMessage(getBytes(emailNoReply))

  val messageNoReply = m.getContent

  val emailReply = scala.io.Source.fromFile("/media/mcapizzi/data/Google_Drive_Arizona/Corpora/enron_full/allen-p/sent/3.").getLines.mkString("\n")

  //doesn't separate message from reply
  val mm = getMessage(getBytes(emailReply))

  val messageReply = mm.getContent



}