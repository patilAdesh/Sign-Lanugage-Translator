# Sign Language Recognition System

A real-time sign language recognition application that translates hand gestures into text and speech, featuring a user-friendly interface and advanced gesture detection capabilities.

## Project Description
This application uses computer vision and machine learning to recognize American Sign Language (ASL) gestures in real-time. It provides a graphical user interface where users can perform sign language gestures in front of their webcam, and the system will translate these gestures into text and speech output.

## Key Features
- Real-time sign language gesture recognition
- Text-to-speech conversion of recognized signs
- User-friendly graphical interface
- Word suggestion system
- Dark/Light mode support
- Debug mode for advanced users
- Trial period with license management

## System Requirements
- Python 3.7 or higher
- Webcam
- Windows operating system
- Minimum 4GB RAM
- GPU support recommended for better performance

## Installation
1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the application:
```bash
python app.py
```
2. Allow the application to access your webcam
3. Position your hand in front of the camera
4. The system will recognize your gestures and display the corresponding text
5. Use the interface controls to:
   - Toggle camera on/off
   - Convert text to speech
   - Clear the text
   - Switch between dark/light modes
   - Access debug information

## Technical Details
- Uses TensorFlow for gesture recognition
- Implements MediaPipe for hand tracking
- PyQt5 for the graphical interface
- OpenCV for video processing
- NLTK for word suggestions
- pyttsx3 for text-to-speech conversion

## Project Structure
- `app.py` - Main application file
- `requirements.txt` - Python dependencies
- `cnn8grps_rad1_model.h5` - Trained model for gesture recognition
- `AtoZ_3.1/` - Dataset directory
- `documentation/` - Additional documentation

## License
This software includes a 30-day trial period. After the trial period, a license key is required to continue using the application.

## Support
For support or licensing inquiries, please contact the developer.

## Credits
Created by Adesh Patil 