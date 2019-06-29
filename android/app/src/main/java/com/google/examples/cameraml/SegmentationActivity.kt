package com.google.examples.cameraml

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.widget.GridView
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import androidx.exifinterface.media.ExifInterface


class SegmentationActivity : AppCompatActivity() {

    private var inputBitmap: Bitmap? = null
    private var imageView: ImageView? = null
    private var colorLegendGrid: GridView? = null
    private var colorLegendAdapter: ColorLegendAdapter? = null
    private var colorLegendList: ArrayList<ColorLegendAdapter.ColorLegend> = ArrayList()
    private var segmentationProcessor: SegmentationProcessor? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_segmentation)
        imageView = findViewById(R.id.image_view)
        colorLegendGrid = findViewById(R.id.color_legend_grid)
        colorLegendAdapter = ColorLegendAdapter(this, colorLegendList)
        colorLegendGrid?.adapter = colorLegendAdapter

        // Receive input bitmap
        val imageFilePath = intent.extras?.getString(INPUT_FILE_KEY)
        if (imageFilePath == null) {
            finish()
            return
        }
        val inputBitmap = loadBitmapFromFile(imageFilePath)
        imageView?.setImageBitmap(inputBitmap)

        // Setup image segmentation engine
        val processor = SegmentationProcessor(this)
        processor
            .initialize()
            .continueWithTask { processor.runSegmentationAsync(inputBitmap) }
            .addOnSuccessListener {
                imageView?.setImageBitmap(it.overlayBitmap)
                setupColorLegend(it.labelList)
            }
            .addOnFailureListener { e -> Log.e(TAG, "Inference failed.", e) }

        this.segmentationProcessor = processor
        this.inputBitmap = inputBitmap
    }

    override fun onDestroy() {
        segmentationProcessor?.close()
        super.onDestroy()
    }

    private fun loadBitmapFromFile(filePath: String): Bitmap {
        val bitmap = BitmapFactory.decodeFile(filePath)

        val exif = ExifInterface(filePath)
        val orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, 1)
        Log.d("EXIF", "Exif: $orientation")
        val matrix = Matrix()
        when (orientation) {
            6 -> matrix.postRotate(90f)
            3 -> matrix.postRotate(180f)
            8 -> matrix.postRotate(270f)
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun setupColorLegend(labelList: Collection<String>) {
        // Generate legends
        labelList.forEach { label ->
            val color = segmentationProcessor?.getLabelToColor(label)
            val colorLegend = ColorLegendAdapter.ColorLegend(label, color!!)
            colorLegendList.add(colorLegend)
        }

        colorLegendAdapter?.notifyDataSetChanged()
    }

    companion object {
        private const val INPUT_FILE_KEY = "imageFilePath"
        private const val TAG = "SegmentationActivity"

        fun newInstance(context: Context, imageFilePath: String): Intent {
            val intent = Intent(context, SegmentationActivity::class.java)
            intent.putExtra(INPUT_FILE_KEY, imageFilePath)
            return intent
        }
    }
}
