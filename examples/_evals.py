GROUND_TRUTH_DATA = {
    'image1.jpg': 'OCEAN',
    'image2.jpg': 'AUDIO',
    'image3.jpg': 'ONT',
    'image4.jpg': 'ACH',
    'image5.jpg': 'MTL',
    'image6.jpg': 'DUSK',
    'image7.jpg': 'DUSK',
    'image8.jpg': 'DUSK',
    'image9.jpg': 'ACH',
    'image10.jpg': 'MTL',
    'image11.jpg': 'IOTX',
    'image12.jpg': 'IOTX',
    'image13.jpg': 'IOTX',
    'image14.jpg': 'DUSK',
    'image15.jpg': 'LQTY'
}

def ocr_evaluation(image_name: str, prediction: str) -> None:
    """Simple function to evaluate prediction with ground truth data."""

    prediction_result = prediction == GROUND_TRUTH_DATA[image_name]
    prediction_result_category = 'Correct' if prediction_result else 'Not Correct'
    
    print(f'Evaluating prediction for {image_name}')
    print(f'Expected: {GROUND_TRUTH_DATA[image_name]} -- Prediction: {prediction}')
    print(f'Result: {prediction_result_category}')
    print('--------------------')