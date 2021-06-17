package org.tensorflow.lite.examples.bertqa;

import android.content.Context;
import androidx.test.platform.app.InstrumentationRegistry;
import java.util.List;
import org.tensorflow.lite.examples.bertqa.lib_interpreter.ml.QaClient;
import org.tensorflow.lite.task.text.qa.BertQuestionAnswerer;

import static com.google.common.truth.Truth.assertThat;

public class BertQaTest {

  private static final String MODEL_PATH = "model.tflite";
  private static final String CONTENT = "Super Bowl 50 was an American football game to " +
      "determine the champion of the National Football League (NFL) for the 2015 season. The " +
      "American Football Conference (AFC) champion Denver Broncos defeated the National Football " +
      "Conference (NFC) champion Carolina Panthers 24\\u201310 to earn their third Super Bowl " +
      "title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco " +
      "Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league " +
      "emphasized the \\\"golden anniversary\\\" with various gold-themed initiatives, as well " +
      "as temporarily suspending the tradition of naming each Super Bowl game with Roman " +
      "numerals (under which the game would have been known as \\\"Super Bowl L\\\"), so that " +
      "the logo could prominently feature the Arabic numerals 50.";
  private static final String QUESTION_TO_ASK = "Where did Super Bowl 50 take place?";
  private static final String ANSWER = "Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.";
  private static final int MAX_NUMBER_OF_CONTENT_TO_TEST = 3;

  private QaClient qaClient;
  private BertQuestionAnswerer questionAnswerer;
  private LoadDatasetClient loadDatasetClient;

  @org.junit.Before
  public void setUp() throws Exception {
    Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();

    // Setup interpreter object.
    qaClient = new QaClient(context, MODEL_PATH);
    qaClient.loadModel();

    // Setup Task Library object.
    questionAnswerer = BertQuestionAnswerer.createFromFile(context, MODEL_PATH);

    // Setup client to load sample content and questions.
    loadDatasetClient = new LoadDatasetClient(context);
  }

  @org.junit.After
  public void tearDown() throws Exception {
    qaClient.unload();
    questionAnswerer.close();
  }

  @org.junit.Test
  public void testInterpreterImplementation() {
    List<org.tensorflow.lite.examples.bertqa.lib_interpreter.ml.QaAnswer> answers =
        qaClient.predict(QUESTION_TO_ASK, CONTENT);
    org.tensorflow.lite.examples.bertqa.lib_interpreter.ml.QaAnswer topAnswer = answers.get(0);

    assertThat(topAnswer.text).isEqualTo(ANSWER);
  }

  @org.junit.Test
  public void testTaskLibraryImplementation() {
    List<org.tensorflow.lite.task.text.qa.QaAnswer> answers =
        questionAnswerer.answer(CONTENT, QUESTION_TO_ASK);
    org.tensorflow.lite.task.text.qa.QaAnswer topAnswer = answers.get(0);
    assertThat(topAnswer.text).isEqualTo(ANSWER);
  }

  @org.junit.Test
  public void testTaskLibraryAndInterpreterReturnSameResult() {
    // Decide how many content example to load and run test to compare results.
    int contentToTestCount = Math.min(
        loadDatasetClient.getTitles().length,
        MAX_NUMBER_OF_CONTENT_TO_TEST
    );

    // Loop through the content examples provided in the sample app.
    for (int i = 0; i<contentToTestCount; i++) {
      // Get each content text and its first example question.
      String content = loadDatasetClient.getContent(i);
      String firstQuestion = loadDatasetClient.getQuestions(i)[0];

      // Get answer by TFLite Task Library.
      List<org.tensorflow.lite.task.text.qa.QaAnswer> answersByTaskLib =
          questionAnswerer.answer(content, firstQuestion);
      org.tensorflow.lite.task.text.qa.QaAnswer topAnswerByTaskLib = answersByTaskLib.get(0);

      // Get answer by TFLite Interpreter.
      List<org.tensorflow.lite.examples.bertqa.lib_interpreter.ml.QaAnswer> answersByInterpreter =
          qaClient.predict(firstQuestion, content);
      org.tensorflow.lite.examples.bertqa.lib_interpreter.ml.QaAnswer topAnswerByInterpreter =
          answersByInterpreter.get(0);

      // Compare the top answer to see whether they are identical.
      assertThat(topAnswerByTaskLib.text).isEqualTo(topAnswerByInterpreter.text);
//      assert topAnswerByTaskLib.text.contentEquals(topAnswerByInterpreter.text);
    }
  }
}