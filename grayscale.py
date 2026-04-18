import cv2

# Read the image
image = cv2.imread("test_image.jpg")

# Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Get info of both
print(f"✅ Original image channels: {image.shape}")
print(f"✅ Grayscale image channels: {gray.shape}")

# Show both images
cv2.imshow("Original", cv2.resize(image, (800, 600)))
cv2.imshow("Grayscale", cv2.resize(gray, (800, 600)))

# Wait for key press
cv2.waitKey(0)
cv2.destroyAllWindows()