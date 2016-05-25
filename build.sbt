import sbtassembly.AssemblyPlugin.autoImport._

name := "NaiveBayes"

version := "1.0"

scalaVersion := "2.11.8"

ivyScala := ivyScala.value map { _.copy(overrideScalaVersion = true) }      //this overrides competing scala versions?  ???

offline := true

assemblyJarName in assembly := "ODIN-sample.jar"

//test in assembly := {}

mainClass in assembly := Some("Main")

assemblyExcludedJars in assembly := {
  val cp = (fullClasspath in assembly).value
  cp filter {_.data.getName == "java-cup-0.11a.jar"}
}

libraryDependencies ++= Seq(
  "org.clulab" %% "processors" % "5.8.2",                                 //required exclude java-cup-0.11a.jar
  "org.clulab" %% "processors" % "5.8.2" classifier "models",
  "org.scalactic" %% "scalactic" % "2.2.6",                               //for unit tests
  "org.scalatest" %% "scalatest" % "2.2.6" % "test",
  "javax.mail" % "mail" % "1.4.7"                                         //for parsing emails
)