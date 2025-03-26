#!/bin/bash

# Create a temporary PNG with pink "Ar" text on transparent background
magick -size 512x512 xc:none -font "Arial-Bold" -pointsize 300 \
       -fill "#ff69b4" -gravity center -draw "text 0,0 'Ar'" \
       temp.png

# Create all the necessary favicon sizes
magick temp.png -resize 180x180 apple-touch-icon.png
magick temp.png -resize 32x32 favicon-32x32.png
magick temp.png -resize 16x16 favicon-16x16.png
magick temp.png -resize 192x192 android-chrome-192x192.png
magick temp.png -resize 512x512 android-chrome-512x512.png

# Create multi-size favicon.ico
magick temp.png -define icon:auto-resize=16,32,48 ../favicon.ico

# Clean up
rm temp.png

echo "Favicons generated successfully!" 