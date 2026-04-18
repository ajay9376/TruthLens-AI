import cv2

# Read the image
image = cv2.imread("test_image.jpg")

# Get image info
height, width, channels = image.shape

print(f"✅ Image loaded successfully!")
print(f"📐 Width: {width}px")
print(f"📐 Height: {height}px")
print(f"🎨 Channels: {channels}")

# Resize image to fit screen
resized = cv2.resize(image, (800, 600))

# Show the resized image
cv2.imshow("TruthLens AI - Test Image", resized)

# Wait until you press any key
cv2.waitKey(0)

# Close the window
cv2.destroyAllWindows()