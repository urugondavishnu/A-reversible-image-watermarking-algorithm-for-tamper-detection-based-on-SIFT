# A-reversible-image-watermarking-algorithm-for-tamper-detection-based-on-SIFT
## Overview
This project implements a reversible image watermarking algorithm designed for tamper detection using the Scale-Invariant Feature Transform (SIFT). The approach ensures that any modifications to the watermarked image can be accurately detected and localized, while also allowing for the original image to be perfectly recovered if no tampering has occurred.

## Features
- **Reversible Watermarking**: Ensures the original image can be restored without any loss after watermark extraction.
- **Tamper Detection**: Accurately identifies and localizes tampered regions within the image.
- **SIFT-Based Embedding**: Utilizes SIFT features for robust and imperceptible watermark embedding.

## Repository Contents
`main.py`: The primary script containing the implementation of the watermarking algorithm.

`baboon.png`, `lena.png`, `plane.png`: Sample images used for testing the algorithm.

`Watermarking_Report.docx`: Detailed report explaining the methodology and results.

`watermark_presentation.pptx`: Presentation slides summarizing the project.

## Prerequisites
Ensure you have the following installed:
- Python 3.x
- OpenCV (cv2)
- NumPy
- Matplotlib
  
You can install the required Python packages using:
  ```bash
  pip install opencv-python numpy matplotlib
```
## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/urugondavishnu/A-reversible-image-watermarking-algorithm-for-tamper-detection-based-on-SIFT.git
2. Navigate to the project directory:
   ```bash
   cd A-reversible-image-watermarking-algorithm-for-tamper-detection-based-on-SIFT
3. Run the main script:
   ```bash
   python main.py
The script will process the sample images, embed the watermark, simulate tampering, and attempt to detect and localize any modifications.

## Methodology
The algorithm operates as follows:
- **Feature Extraction**: The original image is divided into blocks, and SIFT features are extracted from each block.
- **Watermark Generation**: Authentication watermarks are generated based on the extracted features.
- **Embedding**: The watermarks are embedded into the image using a reversible method that ensures the original image can be recovered.
- **Tamper Detection**: Upon receiving a potentially tampered image, the embedded watermarks are extracted and compared against newly computed features to detect discrepancies.
- **Localization**: If tampering is detected, the algorithm localizes the affected regions within the image.

## Results
The implemented method effectively detects and localizes tampered regions in images while maintaining high visual quality and ensuring reversibility. Detailed results and analysis can be found in the `Watermarking_Report.docx` and `watermark_presentation.pptx` files.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your enhancements or bug fixes.

## Contact
For questions or feedback, please reach out to [urugondavishnu](https://github.com/urugondavishnu).
