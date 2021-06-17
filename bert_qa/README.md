# TensorFlow Lite BERT QA Android Example Application

## Overview

This is an end-to-end example of BERT Question & Answer application built with
TensorFlow 2.0, and tested on SQuAD dataset. The demo app provides 48 passages
from the dataset for users to choose from, and gives 5 most possible answers
corresponding to the input passage and query.

These instructions walk you through running the demo on an Android device.

## Run TFLite model with Task Library vs. Interpreter

There are TWO ways to integrate a TFLite BERT Question & Answer model implemented in this Android sample:
1. Using the TFLite Task Library's [BertQuestionAnswerer API](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_question_answerer). This is the easiest way to integrate TFLite models to mobile apps in just a few lines of code.
    * This is the **default** excecution path implemented in the app. If you run the app as-is, the TFLite Task Library will be used to run the TFLite BERT Question & Answer model.
    * The specific code that use the TFLite Task Library is between these comment tags in the sample app.
```
// BEGIN - TFLite Task Library path
...
// END - TFLite Task Library path
```
    
2. Using the TFLite [Interpreter](https://www.tensorflow.org/lite/guide/inference) directly. This allows full customization of the preprocessing/post-processing logic, and can be used if your BERT Question & Answer model requires custom pre/post-processing logic that isn't supported by TFLite Task Library.
    * All code used to integrate the TFLite model using the Interpreter directly is inside the `lib_interpreter` module.
    * This execution path is disabled in the app. You can enable it by commenting out the code of the TFLite Task Library path mentioned above, then enable that of the interpreter path inside these comment tags.
```
// BEGIN - TFLite Interpreter path
...
// END - TFLite Interpreter path
```


## Build the demo using Android Studio

### Prerequisites

*   If you don't have already, install
    [Android Studio](https://developer.android.com/studio/index.html), following
    the instructions on the website.

*   Android Studio 3.2 or later.
    - Gradle 4.6 or higher.
    - SDK Build Tools 28.0.3 or higher.

*   You need an Android device or Android emulator and Android development
    environment with minimum API 15.

### Building

*   Open Android Studio, and from the Welcome screen, select `Open an existing
    Android Studio project`.

*   From the Open File or Project window that appears, navigate to and select
    the `bert_qa/android` directory from wherever you cloned the TensorFlow Lite
    sample GitHub repo.

*   You may also need to install various platforms and tools according to error
    messages.

*   If it asks you to use Instant Run, click Proceed Without Instant Run.

### Running

*   You need to have an Android device plugged in with developer options enabled
    at this point. See [here](https://developer.android.com/studio/run/device)
    for more details on setting up developer devices.

*   If you already have Android emulator installed in Android Studio, select a
    virtual device with minimum API 15.

*   Click `Run` to run the demo app on your Android device.

## Build the demo using gradle (command line)

### Building and Installing

*   Use the following command to build a demo apk:

```
cd lite/examples/bert_qa/android   # Folder for Android app.

./gradlew build
```

*   Use the following command to install the apk onto your connected device:

```
adb install app/build/outputs/apk/debug/app-debug.apk
```
