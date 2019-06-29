package com.google.examples.cameraml

import android.content.Context
import android.view.LayoutInflater
import android.view.View
import android.widget.TextView
import android.view.ViewGroup
import android.widget.BaseAdapter
import android.widget.ImageView


class ColorLegendAdapter
    (private val context: Context, private val colorLegends: List<ColorLegend>) : BaseAdapter() {

    class ColorLegend(val label: String, val color: Int)

    override fun getCount(): Int {
        return colorLegends.size
    }

    override fun getItemId(position: Int): Long {
        return 0
    }

    override fun getItem(position: Int): Any? {
        return null
    }

    override fun getView(position: Int, convertView: View?, parent: ViewGroup?): View {
        val inflater = context.getSystemService(Context.LAYOUT_INFLATER_SERVICE) as LayoutInflater
        val itemView = inflater.inflate(android.R.layout.activity_list_item, null)
        val colorLegend = colorLegends[position]

        itemView.findViewById<TextView>(android.R.id.text1).text = colorLegend.label
        itemView.findViewById<ImageView>(android.R.id.icon).setBackgroundColor(colorLegend.color)

        return itemView
    }

}
