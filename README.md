# Cancer-_Project
Cancer type prediction from Histopathological images.

Cancer type prediction from Histopathological images

AIM:
Develop and predict the specific type of colon and lung cancer based on the distinctive features present in the histopathological images using deep neural networks.

About the dataset:
The dataset involves five classes being:
•	Lung benign tissue
•	Lung adenocarcinoma
•	Lung squamous cell carcinoma
•	Colon adenocarcinoma
•	Colon benign tissue
Dataset contains 25000 images.
•	Dataset size - 5GB
•	One image size: 60 – 100kb
•	All images are 768 x 768 pixels in size and are in jpeg file format.

Workflow:
•	Data Collection
•	Data Pre-processing
•	Splitting the dataset
•	Building and training the model
•	Evaluation






Workflow:
1. Model Definition:
   - Define a CNN model (`ConvolutionalNetwork`) that inherits from the    PyTorch Lightning `LightningModule`.
   - Architecture includes convolutional layers (`conv1`, `conv2`, `conv3`) and fully connected layers (`fc1`, `fc2`, `fc3`, `fc4`).
   - Activation functions (ReLU) are applied after convolutional and fully connected layers.
   - Output layer uses log_softmax activation for classification.

2. Data Preparation:
   - Load and organize the dataset with file paths and corresponding labels.
   - Encode labels using `LabelEncoder` to convert text labels to numerical format.
   - Split the dataset into training and testing sets using `train_test_split`.

3. Image Transformation:
   - Define a series of image transformations using `transforms.Compose` to resize images and convert them to PyTorch tensors.
   - Create a function (`to_tensor`) to load an image, apply transformations, and return the tensor.

4. Create Datasets and Data Loaders:
   - Create PyTorch `TensorDataset` for both training and testing data, combining image tensors and labels.
   - Use PyTorch `DataLoader` to efficiently load batches of data during training and testing.

5. Model Training Configuration:

 
   - Specify training configuration, including the number of training epochs, optimizer (Adam), and learning rate.

6. TensorBoard Logger:
   - Create a TensorBoard logger (`logger`) to log and visualize training progress, metrics, and losses.

7. Training Process:
   - Initialize a PyTorch Lightning `Trainer` with specified configuration, including the logger, hardware accelerator (GPU), and number of devices.
   - Call `trainer.fit(model, train_dataloaders, val_dataloaders)` to start the training process.
   - During training, the model optimizes parameters using backpropagation and updates based on the provided training data.
   - Validation data is used to monitor the model's performance on unseen data.
   - Logs and metrics are recorded and can be visualized using TensorBoard.

8. Post-training Evaluation:
   - After training, the model is ready for evaluation using the test data.
   - Testing data is passed through the trained model, and metrics are logged for evaluation.

9. The workflow demonstrates the end-to-end process of training a CNN for image classification using PyTorch Lightning.
 - The modular structure, combined with PyTorch Lightning features, simplifies the training loop and enhances code readability.

Conclusion:
The proposed project holds significant potential to revolutionize cancer diagnosis by providing medical professionals with a reliable and efficient tool for predicting colon and lung cancer types.

