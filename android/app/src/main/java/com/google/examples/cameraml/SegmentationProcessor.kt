package com.google.examples.cameraml

import android.content.Context
import android.content.res.AssetManager
import android.graphics.*
import android.util.Log
import androidx.annotation.ColorInt
import androidx.lifecycle.LifecycleObserver
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate

import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.*
import java.util.concurrent.Callable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class SegmentationProcessor(private val context: Context) : LifecycleObserver {
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    /** Executor to run inference task in the background */
    private val executorService: ExecutorService = Executors.newCachedThreadPool()

    private var inputImageWidth: Int = 0 // will be inferred from TF Lite model
    private var inputImageHeight: Int = 0 // will be inferred from TF Lite model
    private var modelInputSize: Int = 0 // will be inferred from TF Lite model

    private val labelToColor: MutableMap<String, Int>
    @ColorInt
    private val labelIndexToColor: IntArray

    init {
        // Generate label meta data
        labelToColor = HashMap()
        labelIndexToColor = IntArray(OUTPUT_CLASSES_COUNT)
        for (i in 0 until OUTPUT_CLASSES_COUNT) {
            val color = Color.parseColor(COLOR_STRING_LIST[i])

            // Generate lookup table to convert label string to index
            val label = LABEL_LIST[i]
            labelToColor[label] = color

            // Generate color map for label
            labelIndexToColor[i] = color
        }
    }

    fun initialize(): Task<Void> {
        return Tasks.call(executorService, Callable<Void> {
            initializeInterpreter()
            null
        })
    }

    @Throws(IOException::class)
    private fun initializeInterpreter() {
        val assetManager = context.assets
        val model = loadModelFile(assetManager)

        var options = Interpreter.Options()

        if (USE_GPU) {
            this.gpuDelegate = GpuDelegate()
            options = options.addDelegate(gpuDelegate)
        } else {
            options.setNumThreads(2)
        }
        val interpreter = Interpreter(model, options)

        val inputShape = interpreter.getInputTensor(0).shape()
        inputImageWidth = inputShape[1]
        inputImageHeight = inputShape[2]
        modelInputSize = FLOAT_TYPE_SIZE * inputImageWidth * inputImageHeight * PIXEL_SIZE
        this.interpreter = interpreter
    }

    @Throws(IOException::class)
    private fun loadModelFile(assetManager: AssetManager): ByteBuffer {
        val fileDescriptor = assetManager.openFd(SegmentationProcessor.MODEL_FILE)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun runSegmentation(bitmap: Bitmap): SegmentationResult {
        var startTime: Long
        var elapsedTime: Long

        // Preprocessing: resize the input
        startTime = System.nanoTime()
        val resizedImage = Bitmap.createScaledBitmap(bitmap, inputImageWidth, inputImageHeight, false)
        val byteBuffer = convertBitmapToByteBuffer(resizedImage)
        elapsedTime = (System.nanoTime() - startTime) / 1000000
        Log.d(TAG, "Preprocessing time = " + elapsedTime + "ms")

        startTime = System.nanoTime()
        val rawSegmentationResult =
            Array(1) { Array(inputImageWidth) { Array(inputImageWidth) { FloatArray(OUTPUT_CLASSES_COUNT) } } }
        interpreter?.run(byteBuffer, rawSegmentationResult)
        elapsedTime = (System.nanoTime() - startTime) / 1000000
        Log.d(TAG, "Inference time = " + elapsedTime + "ms")

        startTime = System.nanoTime()
        val segmentation = Array(inputImageWidth) { IntArray(inputImageHeight) }
        val labelList = HashSet<String>()
        convertModelOutputToSegmentationResult(rawSegmentationResult, segmentation, labelList)
        elapsedTime = (System.nanoTime() - startTime) / 1000000
        Log.d(TAG, "Postprocessing time = " + elapsedTime + "ms")

        startTime = System.nanoTime()
        val segBitmap = createOverlayResult(resizedImage, segmentation)
        elapsedTime = (System.nanoTime() - startTime) / 1000000
        Log.d(TAG, "Optional: Result visualization = " + elapsedTime + "ms")

        return SegmentationResult(segBitmap, segmentation, labelList)
    }

    fun runSegmentationAsync(bitmap: Bitmap): Task<SegmentationResult> {
        return Tasks.call(executorService, Callable<SegmentationResult> { runSegmentation(bitmap) })
    }

    fun close() {
        gpuDelegate?.close()
        interpreter?.close()
    }

    fun getLabelToColor(label: String): Int {
        return if (labelToColor.containsKey(label)) {
            labelToColor[label]!!
        } else {
            0
        }
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(modelInputSize)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputImageWidth * inputImageHeight)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        for (pixelValue in pixels) {
            val r = (pixelValue shr 16 and 0xFF) / IMAGE_MEAN - IMAGE_OFFSET
            val g = (pixelValue shr 8 and 0xFF) / IMAGE_MEAN - IMAGE_OFFSET
            val b = (pixelValue and 0xFF) / IMAGE_MEAN - IMAGE_OFFSET
            byteBuffer.putFloat(r)
            byteBuffer.putFloat(g)
            byteBuffer.putFloat(b)
        }

        return byteBuffer
    }

    private fun convertModelOutputToSegmentationResult(
        modelOutput: Array<Array<Array<FloatArray>>>,
        segmentation: Array<IntArray>,
        classList: MutableSet<String>
    ) {
        val width = modelOutput[0].size
        val height = modelOutput[0][0].size

        // Index of segmentation class is index of the max logit
        // Equivalent to tf.argmax(modelOutput, axis = 3)
        val logits = modelOutput[0]

        for (i in 0 until width) {
            for (j in 0 until height) {
                // Find index of largest logit for this pixel
                var max = logits[i][j][0]
                var index = 0

                for (k in 1 until OUTPUT_CLASSES_COUNT) {
                    if (logits[i][j][k] > max) {
                        max = logits[i][j][k]
                        index = k
                    }
                }

                // Set segmentation result
                segmentation[i][j] = index
                classList.add(LABEL_LIST[index])
            }
        }
    }

    private fun createOverlayResult(inputImage: Bitmap, segmentation: Array<IntArray>): Bitmap {
        // Verify if dimensions match
        val width = inputImage.width
        val height = inputImage.height
        if (width != segmentation[0].size || height != segmentation.size) {
            throw IllegalArgumentException("Input image size and segmentation array size does not match " +
                    "($width,$height) != (${segmentation[0].size},${segmentation.size})")
        }

        // Create segmentation bitmap
        val segBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        for (row in 0 until height) {
            for (col in 0 until width) {
                // Note: Bitmap: row-col vs. TFLite input/ouput: col-row
                segBitmap.setPixel(row, col, labelIndexToColor[segmentation[col][row]])
            }
        }

        // Create overlay result
        return overlay(inputImage, segBitmap)
    }

    inner class SegmentationResult internal constructor(
        val overlayBitmap: Bitmap,
        val classOfPixels: Array<IntArray>,
        val labelList: Set<String>
    )

    companion object {
        private const val TAG = "SegmentationProcessor"

        private const val MODEL_FILE = "deeplabv3_257_mv_gpu.tflite"
        //    private static final String MODEL_FILE = "deeplabv3.tflite";

        private const val IMAGE_MEAN = 128.0f
        private const val IMAGE_OFFSET = 1.0f
        private const val FLOAT_TYPE_SIZE = 4
        private const val PIXEL_SIZE = 3
        private const val USE_GPU = false

        private val LABEL_LIST = arrayOf(
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tv"
        )
        private val COLOR_STRING_LIST = arrayOf(
            "black",
            "red",
            "blue",
            "green",
            "gray",
            "cyan",
            "magenta",
            "yellow",
            "grey",
            "aqua",
            "fuchsia",
            "lime",
            "maroon",
            "navy",
            "olive",
            "purple",
            "silver",
            "teal",
            "lightgray",
            "darkgray",
            "white"
        )
        private val OUTPUT_CLASSES_COUNT = LABEL_LIST.size

        private fun overlay(bmp1: Bitmap, bmp2: Bitmap): Bitmap {
            val bmOverlay = Bitmap.createBitmap(bmp1.width, bmp1.height, bmp1.config)
            val canvas = Canvas(bmOverlay)
            canvas.drawBitmap(bmp1, Matrix(), null)

            val alphaPaint = Paint()
            alphaPaint.alpha = 42
            canvas.drawBitmap(bmp2, 0f, 0f, alphaPaint)

            return bmOverlay
        }
    }

}
