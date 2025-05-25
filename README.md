# GeoVision

## Description

GeoVision is a computer vision project focused on predicting geographical information from images. It includes models for estimating direction, latitude/longitude coordinates, and region IDs.

## Subdirectories

-   **direction:** Contains models and scripts for predicting the direction (e.g., North, South, East, West) from an image.
    -   `train_a.py`: Script for training the direction prediction model.
    -   `test_a.py`: Script for testing the direction prediction model.
    -   `README.md`: README file with details about the direction prediction model.
-   **latlong:** Contains models and scripts for predicting the latitude and longitude coordinates from an image.
    -   `train_l.py`: Script for training the latitude/longitude prediction model.
    -   `test_l.py`: Script for testing the latitude/longitude prediction model.
    -   `README.md`: README file with details about the latitude/longitude prediction model.
-   **region\_id:** Contains models and scripts for predicting the region ID from an image.
    -   `train_r.py`: Script for training the region ID prediction model.
    -   `test_r.py`: Script for testing the region ID prediction model.
    -   `README.md`: README file with details about the region ID prediction model.

## Usage

Each subdirectory contains scripts for training and testing the corresponding model. To use the models:

1.  Navigate to the desired subdirectory (e.g., `cd GeoVision/direction`).
2.  Train the model using the `train_a.py` script:

    ```bash
    python train_a.py
    ```

3.  Test the model using the `test_a.py` script:

    ```bash
    python test_a.py
    ```

Refer to the README file in each subdirectory for more specific instructions and details about the models and datasets used.

## Contributing

Contributions are welcome! If you find a bug or have a suggestion, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
