import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                           QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                           QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                           QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                           QDialog, QLineEdit, QFormLayout, QRadioButton,
                           QButtonGroup)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn import datasets, preprocessing, model_selection
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, mean_squared_error, mean_absolute_error, 
                           confusion_matrix, r2_score, roc_curve, auc, 
                           precision_recall_curve, f1_score)
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI")
        self.setGeometry(100, 100, 1400, 800)
        
        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        
        # Initialize data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        self.posterior_probs = None
        
        # Neural network configuration
        self.layer_config = []
        
        # Create components
        self.create_data_section()
        self.create_tabs()
        self.create_visualization()
        self.create_status_bar()
    
    def load_dataset(self):
        """Load selected dataset"""
        try:
            dataset_name = self.dataset_combo.currentText()
            
            if dataset_name == "Load Custom Dataset":
                return
            
            # Load selected dataset
            if dataset_name == "Iris Dataset":
                data = datasets.load_iris()
                self.X = data.data
                self.y = data.target
            elif dataset_name == "Breast Cancer Dataset":
                data = datasets.load_breast_cancer()
                self.X = data.data
                self.y = data.target
            elif dataset_name == "Digits Dataset":
                data = datasets.load_digits()
                self.X = data.data
                self.y = data.target
            elif dataset_name == "Boston Housing Dataset":
                data = datasets.fetch_california_housing() 
                self.X = data.data
                self.y = data.target
            elif dataset_name == "MNIST Dataset":
                (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
                self.X_train, self.X_test = X_train, X_test
                self.y_train, self.y_test = y_train, y_test
                self.status_bar.showMessage(f"Loaded {dataset_name}")
                return
            
            # Apply missing value handling if needed
            self.apply_missing_value_handling(self.X)
            
            # Split data
            test_size = self.split_spin.value()
            self.X_train, self.X_test, self.y_train, self.y_test = \
                model_selection.train_test_split(self.X, self.y, 
                                              test_size=test_size, 
                                              random_state=42)
            
            # Apply scaling if selected
            self.apply_scaling()
            
            self.status_bar.showMessage(f"Loaded {dataset_name}")
            
        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")
    
    def load_custom_data(self):
        """Load custom dataset from CSV file"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Dataset",
                "",
                "CSV files (*.csv)"
            )
            
            if file_name:
                # Load data
                data = pd.read_csv(file_name)
                
                # Ask user to select target column
                target_col = self.select_target_column(data.columns)
                
                if target_col:
                    X = data.drop(target_col, axis=1)
                    y = data[target_col]
                    
                    # Apply missing value handling
                    X = self.apply_missing_value_handling(X)
                    
                    # Split data
                    test_size = self.split_spin.value()
                    self.X_train, self.X_test, self.y_train, self.y_test = \
                        model_selection.train_test_split(X, y, 
                                                      test_size=test_size, 
                                                      random_state=42)
                    
                    # Apply scaling if selected
                    self.apply_scaling()
                    
                    self.status_bar.showMessage(f"Loaded custom dataset: {file_name}")
                    
        except Exception as e:
            self.show_error(f"Error loading custom dataset: {str(e)}")
    
    def select_target_column(self, columns):
        """Dialog to select target column from dataset"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Target Column")
        layout = QVBoxLayout(dialog)
        
        combo = QComboBox()
        combo.addItems(columns)
        layout.addWidget(combo)
        
        btn = QPushButton("Select")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return combo.currentText()
        return None
    
    def apply_missing_value_handling(self, X):
        """Apply selected missing value handling method to data"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        imputation_method = self.imputation_combo.currentText()
        
        if imputation_method != "No Imputation":
            try:
                if imputation_method == "Mean Imputation":
                    imputer = SimpleImputer(strategy='mean')
                    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
                    
                elif imputation_method == "Median Imputation":
                    imputer = SimpleImputer(strategy='median')
                    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
                    
                elif imputation_method == "Most Frequent Imputation":
                    imputer = SimpleImputer(strategy='most_frequent')
                    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
                    
                elif imputation_method == "Forward Fill":
                    X = X.fillna(method='ffill')
                    
                elif imputation_method == "Backward Fill":
                    X = X.fillna(method='bfill')
                    
                elif imputation_method == "Linear Interpolation":
                    X = X.interpolate(method='linear')
                
            except Exception as e:
                self.show_error(f"Error applying imputation: {str(e)}")
        
        return X
    
    def apply_scaling(self):
        """Apply selected scaling method to the data"""
        scaling_method = self.scaling_combo.currentText()
        
        if scaling_method != "No Scaling":
            try:
                if scaling_method == "Standard Scaling":
                    scaler = preprocessing.StandardScaler()
                elif scaling_method == "Min-Max Scaling":
                    scaler = preprocessing.MinMaxScaler()
                elif scaling_method == "Robust Scaling":
                    scaler = preprocessing.RobustScaler()
                
                self.X_train = scaler.fit_transform(self.X_train)
                self.X_test = scaler.transform(self.X_test)
                
            except Exception as e:
                self.show_error(f"Error applying scaling: {str(e)}")
    
    def create_data_section(self):
        """Create the data loading and preprocessing section"""
        data_group = QGroupBox("Data Management")
        data_layout = QGridLayout()
        
        # Row 1: Dataset selection and loading
        data_layout.addWidget(QLabel("Dataset:"), 0, 0)
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Load Custom Dataset",
            "Iris Dataset",
            "Breast Cancer Dataset",
            "Digits Dataset",
            "Boston Housing Dataset",
            "MNIST Dataset"
        ])
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)
        data_layout.addWidget(self.dataset_combo, 0, 1)
        
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_custom_data)
        data_layout.addWidget(self.load_btn, 0, 2)
        
        # Row 2: Preprocessing options
        data_layout.addWidget(QLabel("Scaling:"), 1, 0)
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems([
            "No Scaling",
            "Standard Scaling",
            "Min-Max Scaling",
            "Robust Scaling"
        ])
        data_layout.addWidget(self.scaling_combo, 1, 1)
        
        data_layout.addWidget(QLabel("Test Split:"), 1, 2)
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setValue(0.2)
        self.split_spin.setSingleStep(0.1)
        data_layout.addWidget(self.split_spin, 1, 3)
        
        # Row 3: Missing value handling
        data_layout.addWidget(QLabel("Missing Value Handling:"), 2, 0)
        self.imputation_combo = QComboBox()
        self.imputation_combo.addItems([
            "No Imputation",
            "Mean Imputation",
            "Median Imputation",
            "Most Frequent Imputation",
            "Forward Fill",
            "Backward Fill",
            "Linear Interpolation"
        ])
        data_layout.addWidget(self.imputation_combo, 2, 1)
        
        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)
    
    def create_tabs(self):
        """Create tabs for different ML topics"""
        self.tab_widget = QTabWidget()
        
        # Create individual tabs
        tabs = [
            ("Classical ML", self.create_classical_ml_tab),
            ("Deep Learning", self.create_deep_learning_tab),
            ("Dimensionality Reduction", self.create_dim_reduction_tab),
            ("Reinforcement Learning", self.create_rl_tab)
        ]
        
        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)
        
        self.layout.addWidget(self.tab_widget)
    
    def create_classical_ml_tab(self):
        """Create the classical machine learning algorithms tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Regression section
        regression_group = QGroupBox("Regression")
        regression_layout = QVBoxLayout()
        
        # Linear Regression
        lr_group = self.create_algorithm_group(
            "Linear Regression",
            {"fit_intercept": "checkbox"},
            loss_options=["MSE", "MAE"]
        )
        regression_layout.addWidget(lr_group)
        
        # Support Vector Regression (SVR)
        svr_group = self.create_algorithm_group(
            "Support Vector Regression",
            {"C": "double",
             "epsilon": "double",
             "kernel": ["linear", "rbf", "poly"]},
            loss_options=["MSE", "MAE", "Huber"]
        )
        regression_layout.addWidget(svr_group)
        
        regression_group.setLayout(regression_layout)
        layout.addWidget(regression_group, 0, 0)
        
        # Classification section
        classification_group = QGroupBox("Classification")
        classification_layout = QVBoxLayout()
        
        # Logistic Regression
        logistic_group = self.create_algorithm_group(
            "Logistic Regression",
            {"C": "double",
             "max_iter": "int",
             "multi_class": ["ovr", "multinomial"]},
            loss_options=["Cross-Entropy", "Hinge"]
        )
        classification_layout.addWidget(logistic_group)
        
        # Support Vector Machine (SVM)
        svm_group = self.create_algorithm_group(
            "Support Vector Machine",
            {"C": "double",
             "kernel": ["linear", "rbf", "poly"],
             "degree": "int",
             "gamma": ["scale", "auto"]},
            loss_options=["Hinge", "Cross-Entropy"]
        )
        classification_layout.addWidget(svm_group)
        
        # Decision Trees
        dt_group = self.create_algorithm_group(
            "Decision Tree",
            {"max_depth": "int",
             "min_samples_split": "int",
             "criterion": ["gini", "entropy"]},
            loss_options=["Gini", "Entropy"]
        )
        classification_layout.addWidget(dt_group)
        
        # Random Forest
        rf_group = self.create_algorithm_group(
            "Random Forest",
            {"n_estimators": "int",
             "max_depth": "int",
             "min_samples_split": "int"},
            loss_options=["Gini", "Entropy"]
        )
        classification_layout.addWidget(rf_group)
        
        # KNN
        knn_group = self.create_algorithm_group(
            "K-Nearest Neighbors",
            {"n_neighbors": "int",
             "weights": ["uniform", "distance"],
             "metric": ["euclidean", "manhattan"]},
            loss_options=None  # KNN doesn't use a loss function in the same way
        )
        classification_layout.addWidget(knn_group)
        
        classification_group.setLayout(classification_layout)
        layout.addWidget(classification_group, 0, 1)
        
        # Bayesian Methods section
        bayesian_group = QGroupBox("Bayesian Methods")
        bayesian_layout = QVBoxLayout()
        
        # Gaussian Naive Bayes
        gnb_group = self.create_bayesian_group()
        bayesian_layout.addWidget(gnb_group)
        
        bayesian_group.setLayout(bayesian_layout)
        layout.addWidget(bayesian_group, 1, 0, 1, 2)
        
        return widget
    
    def create_bayesian_group(self):
        """Create Gaussian Naive Bayes configuration group"""
        group = QGroupBox("Gaussian Naive Bayes")
        layout = QVBoxLayout()
        
        # Var smoothing parameter
        var_layout = QHBoxLayout()
        var_layout.addWidget(QLabel("Var Smoothing:"))
        self.var_smoothing = QDoubleSpinBox()
        self.var_smoothing.setRange(1e-12, 1.0)
        self.var_smoothing.setValue(1e-9)
        self.var_smoothing.setDecimals(12)
        self.var_smoothing.setSingleStep(1e-10)
        var_layout.addWidget(self.var_smoothing)
        layout.addLayout(var_layout)
        
        # Prior probabilities
        prior_layout = QVBoxLayout()
        prior_layout.addWidget(QLabel("Prior Probabilities:"))
        
        self.prior_type_group = QButtonGroup()
        
        self.uniform_prior = QRadioButton("Uniform")
        self.uniform_prior.setChecked(True)
        self.prior_type_group.addButton(self.uniform_prior)
        prior_layout.addWidget(self.uniform_prior)
        
        self.custom_prior = QRadioButton("Custom")
        self.prior_type_group.addButton(self.custom_prior)
        prior_layout.addWidget(self.custom_prior)
        
        # Custom prior input
        custom_prior_layout = QHBoxLayout()
        custom_prior_layout.addWidget(QLabel("Custom Priors (comma-separated):"))
        self.custom_prior_input = QLineEdit()
        self.custom_prior_input.setPlaceholderText("e.g., 0.3,0.7 for binary classification")
        custom_prior_layout.addWidget(self.custom_prior_input)
        prior_layout.addLayout(custom_prior_layout)
        
        layout.addLayout(prior_layout)
        
        # Display posterior checkbox
        self.show_posterior = QCheckBox("Show Posterior Probabilities in Visualization")
        self.show_posterior.setChecked(True)
        layout.addWidget(self.show_posterior)
        
        # Train button
        train_btn = QPushButton("Train Gaussian Naive Bayes")
        train_btn.clicked.connect(self.train_bayesian_model)
        layout.addWidget(train_btn)
        
        group.setLayout(layout)
        return group
    
    def train_bayesian_model(self):
        """Train a Gaussian Naive Bayes model with current configuration"""
        if self.X_train is None or self.y_train is None:
            self.show_error("Please load data first")
            return
        
        try:
            # Get configuration
            var_smoothing = self.var_smoothing.value()
            
            # Get priors
            priors = None
            if self.custom_prior.isChecked():
                try:
                    priors_text = self.custom_prior_input.text()
                    if priors_text:
                        priors = [float(x.strip()) for x in priors_text.split(',')]
                        # Check if priors sum to 1
                        if abs(sum(priors) - 1.0) > 1e-6:
                            self.show_error("Custom priors must sum to 1.0")
                            return
                except Exception as e:
                    self.show_error(f"Invalid custom priors: {str(e)}")
                    return
            
            # Create and train model
            model = GaussianNB(var_smoothing=var_smoothing, priors=priors)
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Store posterior probabilities for visualization
            if self.show_posterior.isChecked():
                self.posterior_probs = model.predict_proba(self.X_test)
            else:
                self.posterior_probs = None
            
            # Update current model
            self.current_model = model
            
            # Update visualization
            self.update_visualization(y_pred)
            self.update_metrics(y_pred)
            
            self.status_bar.showMessage("Gaussian Naive Bayes training complete")
            
        except Exception as e:
            self.show_error(f"Error training Gaussian Naive Bayes: {str(e)}")
    
    def create_dim_reduction_tab(self):
        """Create the dimensionality reduction tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # K-Means section
        kmeans_group = QGroupBox("K-Means Clustering")
        kmeans_layout = QVBoxLayout()
        
        kmeans_params = self.create_algorithm_group(
            "K-Means Parameters",
            {"n_clusters": "int",
             "max_iter": "int",
             "n_init": "int"}
        )
        kmeans_layout.addWidget(kmeans_params)
        
        kmeans_group.setLayout(kmeans_layout)
        layout.addWidget(kmeans_group, 0, 0)
        
        # PCA section
        pca_group = QGroupBox("Principal Component Analysis")
        pca_layout = QVBoxLayout()
        
        pca_params = self.create_algorithm_group(
            "PCA Parameters",
            {"n_components": "int",
             "whiten": "checkbox"}
        )
        pca_layout.addWidget(pca_params)
        
        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group, 0, 1)
        
        return widget
    
    def create_deep_learning_tab(self):
        """Create the deep learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # MLP section
        mlp_group = QGroupBox("Multi-Layer Perceptron")
        mlp_layout = QVBoxLayout()
        
        # Layer configuration
        self.layer_config = []
        layer_btn = QPushButton("Add Layer")
        layer_btn.clicked.connect(self.add_layer_dialog)
        mlp_layout.addWidget(layer_btn)
        
        # Training parameters
        training_params_group = self.create_training_params_group()
        mlp_layout.addWidget(training_params_group)
        
        # Loss function selection
        loss_layout = QHBoxLayout()
        loss_layout.addWidget(QLabel("Loss Function:"))
        self.dl_loss_combo = QComboBox()
        self.dl_loss_combo.addItems([
            "Cross-Entropy", "MSE", "MAE", "Hinge"
        ])
        loss_layout.addWidget(self.dl_loss_combo)
        mlp_layout.addLayout(loss_layout)
        
        # Train button
        train_btn = QPushButton("Train Neural Network")
        train_btn.clicked.connect(self.train_neural_network)
        mlp_layout.addWidget(train_btn)
        
        mlp_group.setLayout(mlp_layout)
        layout.addWidget(mlp_group, 0, 0)
        
        # CNN section
        cnn_group = QGroupBox("Convolutional Neural Network")
        cnn_layout = QVBoxLayout()
        
        # CNN architecture controls
        cnn_controls = self.create_cnn_controls()
        cnn_layout.addWidget(cnn_controls)
        
        cnn_group.setLayout(cnn_layout)
        layout.addWidget(cnn_group, 0, 1)
        
        # RNN section
        rnn_group = QGroupBox("Recurrent Neural Network")
        rnn_layout = QVBoxLayout()
        
        # RNN architecture controls
        rnn_controls = self.create_rnn_controls()
        rnn_layout.addWidget(rnn_controls)
        
        rnn_group.setLayout(rnn_layout)
        layout.addWidget(rnn_group, 1, 0)
        
        return widget
    
    def create_rl_tab(self):
        """Create the reinforcement learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Environment selection
        env_group = QGroupBox("Environment")
        env_layout = QVBoxLayout()
        
        self.env_combo = QComboBox()
        self.env_combo.addItems([
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1"
        ])
        env_layout.addWidget(self.env_combo)
        
        env_group.setLayout(env_layout)
        layout.addWidget(env_group, 0, 0)
        
        # RL Algorithm selection
        algo_group = QGroupBox("RL Algorithm")
        algo_layout = QVBoxLayout()
        
        self.rl_algo_combo = QComboBox()
        self.rl_algo_combo.addItems([
            "Q-Learning",
            "SARSA",
            "DQN"
        ])
        algo_layout.addWidget(self.rl_algo_combo)
        
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group, 0, 1)
        
        return widget
    
    def create_visualization(self):
        """Create the visualization section"""
        viz_group = QGroupBox("Visualization")
        viz_layout = QHBoxLayout()
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)
        
        # Metrics display
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        viz_layout.addWidget(self.metrics_text)
        
        viz_group.setLayout(viz_layout)
        self.layout.addWidget(viz_group)
    
    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def create_algorithm_group(self, name, params, loss_options=None):
        """Helper method to create algorithm parameter groups with loss function selection"""
        group = QGroupBox(name)
        layout = QVBoxLayout()
        
        # Create parameter inputs
        param_widgets = {}
        for param_name, param_type in params.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param_name}:"))
            
            if param_type == "int":
                widget = QSpinBox()
                widget.setRange(1, 1000)
                if param_name == "n_estimators":
                    widget.setValue(100)
                elif param_name == "max_depth":
                    widget.setValue(5)
                elif param_name == "min_samples_split":
                    widget.setValue(2)
                elif param_name == "n_neighbors":
                    widget.setValue(5)
                elif param_name == "max_iter":
                    widget.setValue(100)
                else:
                    widget.setValue(3)
            elif param_type == "double":
                widget = QDoubleSpinBox()
                widget.setRange(0.0001, 1000.0)
                widget.setSingleStep(0.1)
                if param_name == "C":
                    widget.setValue(1.0)
                elif param_name == "epsilon":
                    widget.setValue(0.1)
                else:
                    widget.setValue(0.5)
            elif param_type == "checkbox":
                widget = QCheckBox()
            elif isinstance(param_type, list):
                widget = QComboBox()
                widget.addItems(param_type)
            
            param_layout.addWidget(widget)
            param_widgets[param_name] = widget
            layout.addLayout(param_layout)
        
        # Add loss function selection if provided
        if loss_options:
            loss_layout = QHBoxLayout()
            loss_layout.addWidget(QLabel("Loss Function:"))
            
            loss_combo = QComboBox()
            loss_combo.addItems(loss_options)
            loss_layout.addWidget(loss_combo)
            
            layout.addLayout(loss_layout)
            param_widgets["loss_function"] = loss_combo
        
        # Add train button
        train_btn = QPushButton(f"Train {name}")
        train_btn.clicked.connect(lambda: self.train_model(name, param_widgets))
        layout.addWidget(train_btn)
        
        group.setLayout(layout)
        return group
    
    def create_training_params_group(self):
        """Create group for neural network training parameters"""
        group = QGroupBox("Training Parameters")
        layout = QVBoxLayout()
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1000)
        self.batch_size_spin.setValue(32)
        batch_layout.addWidget(self.batch_size_spin)
        layout.addLayout(batch_layout)
        
        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        epochs_layout.addWidget(self.epochs_spin)
        layout.addLayout(epochs_layout)
        
        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.001)
        lr_layout.addWidget(self.lr_spin)
        layout.addLayout(lr_layout)
        
        group.setLayout(layout)
        return group
    
    def create_cnn_controls(self):
        """Create controls for Convolutional Neural Network"""
        group = QGroupBox("CNN Architecture")
        layout = QVBoxLayout()
        
        # Placeholder for CNN-specific controls
        label = QLabel("CNN Controls (To be implemented)")
        layout.addWidget(label)
        
        group.setLayout(layout)
        return group
    
    def create_rnn_controls(self):
        """Create controls for Recurrent Neural Network"""
        group = QGroupBox("RNN Architecture")
        layout = QVBoxLayout()
        
        # Placeholder for RNN-specific controls
        label = QLabel("RNN Controls (To be implemented)")
        layout.addWidget(label)
        
        group.setLayout(layout)
        return group
    
    def add_layer_dialog(self):
        """Open a dialog to add a neural network layer"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Neural Network Layer")
        layout = QVBoxLayout(dialog)
        
        # Layer type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        type_combo.addItems(["Dense", "Conv2D", "MaxPooling2D", "Flatten", "Dropout"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)
        
        # Parameters input
        params_group = QGroupBox("Layer Parameters")
        params_layout = QVBoxLayout()
        
        # Dynamic parameter inputs based on layer type
        self.layer_param_inputs = {}
        
        def update_params():
            # Clear existing parameter inputs
            for widget in list(self.layer_param_inputs.values()):
                params_layout.removeWidget(widget)
                widget.deleteLater()
            self.layer_param_inputs.clear()
            
            layer_type = type_combo.currentText()
            if layer_type == "Dense":
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1000)
                units_input.setValue(32)
                self.layer_param_inputs["units"] = units_input
                
                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh", "softmax"])
                self.layer_param_inputs["activation"] = activation_combo
                
                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)
            
            elif layer_type == "Conv2D":
                filters_label = QLabel("Filters:")
                filters_input = QSpinBox()
                filters_input.setRange(1, 1000)
                filters_input.setValue(32)
                self.layer_param_inputs["filters"] = filters_input
                
                kernel_label = QLabel("Kernel Size:")
                kernel_input = QLineEdit()
                kernel_input.setText("3, 3")
                self.layer_param_inputs["kernel_size"] = kernel_input
                
                params_layout.addWidget(filters_label)
                params_layout.addWidget(filters_input)
                params_layout.addWidget(kernel_label)
                params_layout.addWidget(kernel_input)
            
            elif layer_type == "Dropout":
                rate_label = QLabel("Dropout Rate:")
                rate_input = QDoubleSpinBox()
                rate_input.setRange(0.0, 1.0)
                rate_input.setValue(0.5)
                rate_input.setSingleStep(0.1)
                self.layer_param_inputs["rate"] = rate_input
                
                params_layout.addWidget(rate_label)
                params_layout.addWidget(rate_input)
        
        type_combo.currentIndexChanged.connect(update_params)
        update_params()  # Initial update
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        def add_layer():
            layer_type = type_combo.currentText()
            
            # Collect parameters
            layer_params = {}
            for param_name, widget in self.layer_param_inputs.items():
                if isinstance(widget, QSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    layer_params[param_name] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    # Handle kernel size or other tuple-like inputs
                    if param_name == "kernel_size":
                        layer_params[param_name] = tuple(map(int, widget.text().split(',')))
            
            self.layer_config.append({
                "type": layer_type,
                "params": layer_params
            })
            
            dialog.accept()
        
        add_btn.clicked.connect(add_layer)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec()
    
    def train_model(self, model_name, param_widgets):
        """Train selected model with current parameters"""
        if self.X_train is None or self.y_train is None:
            self.show_error("Please load data first")
            return
        
        try:
            # Extract parameters from widgets
            params = {}
            for param_name, widget in param_widgets.items():
                if param_name == "loss_function":
                    continue  # Handle loss function separately
                
                if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                    params[param_name] = widget.value()
                elif isinstance(widget, QCheckBox):
                    params[param_name] = widget.isChecked()
                elif isinstance(widget, QComboBox):
                    params[param_name] = widget.currentText()
            
            # Get loss function if available
            loss_function = None
            if "loss_function" in param_widgets:
                loss_function = param_widgets["loss_function"].currentText()
            
            # Create and train model based on selection
            if model_name == "Linear Regression":
                # For standard linear regression (MSE loss is default)
                if loss_function == "MAE":
                    # For MAE loss, use a more stable approach with Huber loss
                    from sklearn.linear_model import HuberRegressor
                    
                    # Huber is a compromise between MSE and MAE
                    # Lower epsilon makes it closer to MAE
                    try:
                        model = HuberRegressor(epsilon=1.2, max_iter=1000, alpha=0.0)
                        model.fit(self.X_train, self.y_train)
                    except Exception:
                        # Fallback to standard LinearRegression if HuberRegressor fails
                        self.show_error("HuberRegressor failed, falling back to standard LinearRegression")
                        model = LinearRegression(**params)
                        model.fit(self.X_train, self.y_train)
                else:  # Default MSE loss
                    model = LinearRegression(**params)
                    model.fit(self.X_train, self.y_train)
                    
                y_pred = model.predict(self.X_test)
                
            elif model_name == "Logistic Regression":
                # LogisticRegression doesn't accept 'loss' parameter directly
                # For Hinge loss, we should use LinearSVC instead
                if loss_function == "Hinge":
                    from sklearn.svm import LinearSVC
                    # Store original params for LogisticRegression that are compatible with LinearSVC
                    svc_params = {k: v for k, v in params.items() 
                                 if k in ['C', 'max_iter']}
                    model = LinearSVC(**svc_params, dual=False)
                else:  # Cross-Entropy (default for LogisticRegression)
                    model = LogisticRegression(**params)
                
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                
            elif model_name == "Support Vector Machine":
                # Apply loss function
                if "gamma" in params and params["gamma"] in ["scale", "auto"]:
                    params["gamma"] = params["gamma"]  # Keep as string
                
                model = SVC(**params)
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                
            elif model_name == "Support Vector Regression":
                # Handle different kernel configurations properly
                kernel = params.pop("kernel", "rbf")  # Default to rbf if not specified
                
                # Create SVR with appropriate parameters
                try:
                    model = SVR(kernel=kernel, **params)
                    model.fit(self.X_train, self.y_train)
                    y_pred = model.predict(self.X_test)
                except Exception as e:
                    self.show_error(f"Error with SVR: {str(e)}")
                    return
                
            elif model_name == "Naive Bayes":
                model = GaussianNB(**params)
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier(**params)
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                
            elif model_name == "Random Forest":
                model = RandomForestClassifier(**params)
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                
            elif model_name == "K-Nearest Neighbors":
                model = KNeighborsClassifier(**params)
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                
            elif model_name == "K-Means Parameters":
                # Handle K-Means clustering differently
                model = KMeans(**params)
                y_pred = model.fit_predict(self.X_test)
                
            elif model_name == "PCA Parameters":
                # Handle PCA differently
                model = PCA(**params)
                transformed_data = model.fit_transform(self.X_test)
                
                # Use transformed data for visualization
                self.plot_pca_results(transformed_data, model)
                self.status_bar.showMessage("PCA Complete")
                return
            
            # Update current model
            self.current_model = model
            self.posterior_probs = None  # Reset posterior probabilities
            
            # Update visualization
            self.update_visualization(y_pred)
            self.update_metrics(y_pred)
            
            self.status_bar.showMessage(f"{model_name} Training Complete")
            
        except Exception as e:
            self.show_error(f"Error training {model_name}: {str(e)}")
    
    def train_neural_network(self):
        """Train the neural network with current configuration"""
        if not self.layer_config:
            self.show_error("Please add at least one layer to the network")
            return
        
        try:
            # Create and compile model
            model = self.create_neural_network()
            
            # Get training parameters
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()
            
            # Get loss function
            loss_function_name = self.dl_loss_combo.currentText()
            
            # Map loss function name to Keras loss function
            if loss_function_name == "Cross-Entropy":
                loss_function = losses.SparseCategoricalCrossentropy()
            elif loss_function_name == "MSE":
                loss_function = losses.MeanSquaredError()
            elif loss_function_name == "MAE":
                loss_function = losses.MeanAbsoluteError()
            elif loss_function_name == "Hinge":
                loss_function = losses.Hinge()
            
            # Prepare data for neural network
            X_train = self.X_train
            X_test = self.X_test
            
            # Reshape input if needed (e.g., for MNIST)
            if len(X_train.shape) == 3:  # For image data like MNIST
                # Add channel dimension
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
            
            # Use sparse categorical cross-entropy for integer labels
            # otherwise convert to one-hot encoding
            if loss_function_name == "Cross-Entropy":
                y_train = self.y_train
                y_test = self.y_test
            else:
                # One-hot encode target for other loss functions
                y_train = tf.keras.utils.to_categorical(self.y_train)
                y_test = tf.keras.utils.to_categorical(self.y_test)
            
            # Compile model
            optimizer = optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer,
                          loss=loss_function,
                          metrics=['accuracy'])
            
            # Train model
            self.progress_bar.setValue(0)
            history = model.fit(X_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_data=(X_test, y_test),
                                callbacks=[self.create_progress_callback()])
            
            # Update visualization with training history
            self.plot_training_history(history)
            
            # Save current model
            self.current_model = model
            
            self.status_bar.showMessage("Neural Network Training Complete")
            
        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")
    
    def create_neural_network(self):
        """Create neural network based on current configuration"""
        model = models.Sequential()
        
        # Add layers based on configuration
        for i, layer_config in enumerate(self.layer_config):
            layer_type = layer_config["type"]
            params = layer_config["params"]
            
            if layer_type == "Dense":
                # Add input shape for the first layer
                if i == 0:
                    if len(self.X_train.shape) == 1:  # 1D input
                        params['input_shape'] = (1,)
                    else:
                        params['input_shape'] = (self.X_train.shape[1],)
                model.add(layers.Dense(**params))
            elif layer_type == "Conv2D":
                # Add input shape for the first layer
                if i == 0:
                    if len(self.X_train.shape) == 3:  # Image data
                        params['input_shape'] = (self.X_train.shape[1], self.X_train.shape[2], 1)
                    else:
                        # Reshape data to 2D if it's not already
                        dim = int(np.sqrt(self.X_train.shape[1]))
                        params['input_shape'] = (dim, dim, 1)
                model.add(layers.Conv2D(**params))
            elif layer_type == "MaxPooling2D":
                model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            elif layer_type == "Flatten":
                model.add(layers.Flatten())
            elif layer_type == "Dropout":
                model.add(layers.Dropout(**params))
        
        # Check if final layer has softmax activation
        has_softmax = False
        if self.layer_config and self.layer_config[-1]["type"] == "Dense":
            if "activation" in self.layer_config[-1]["params"]:
                has_softmax = self.layer_config[-1]["params"]["activation"] == "softmax"
        
        # Add output layer if final layer doesn't have softmax
        if not has_softmax:
            # Count unique classes for classification
            num_classes = len(np.unique(self.y_train))
            
            if num_classes > 1:  # Classification task
                model.add(layers.Dense(num_classes, activation='softmax'))
            else:  # Regression task
                model.add(layers.Dense(1))
        
        return model
    
    def create_progress_callback(self):
        """Create callback for updating progress bar during training"""
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, progress_bar):
                super().__init__()
                self.progress_bar = progress_bar
                
            def on_epoch_end(self, epoch, logs=None):
                progress = int(((epoch + 1) / self.params['epochs']) * 100)
                self.progress_bar.setValue(progress)
                
        return ProgressCallback(self.progress_bar)
    
    def update_visualization(self, y_pred):
        """Update the visualization with current results"""
        self.figure.clear()
        
        # Check if we're showing posterior probabilities for Naive Bayes
        if self.posterior_probs is not None:
            self.plot_posterior_probabilities()
            return
        
        # Create appropriate visualization based on data
        if len(np.unique(self.y_test)) > 10:  # Regression
            ax = self.figure.add_subplot(111)
            ax.scatter(self.y_test, y_pred)
            ax.plot([self.y_test.min(), self.y_test.max()],
                   [self.y_test.min(), self.y_test.max()],
                   'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Regression Results")
            
        else:  # Classification
            if self.X_test.shape[1] > 2:  # Use PCA for visualization
                pca = PCA(n_components=2)
                X_test_2d = pca.fit_transform(self.X_test)
                
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1],
                                   c=y_pred, cmap='viridis')
                ax.set_xlabel("PCA Component 1")
                ax.set_ylabel("PCA Component 2")
                ax.set_title("Classification Results (PCA Visualization)")
                self.figure.colorbar(scatter)
                
            else:  # Direct 2D visualization
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(self.X_test[:, 0], self.X_test[:, 1],
                                   c=y_pred, cmap='viridis')
                ax.set_xlabel("Feature 1")
                ax.set_ylabel("Feature 2")
                ax.set_title("Classification Results")
                self.figure.colorbar(scatter)
        
        self.canvas.draw()
    
    def plot_posterior_probabilities(self):
        """Plot posterior probabilities for Naive Bayes"""
        ax = self.figure.add_subplot(111)
        
        # Get number of classes
        n_classes = self.posterior_probs.shape[1]
        
        # Plot histograms of posterior probabilities for each class
        for i in range(n_classes):
            ax.hist(self.posterior_probs[:, i], alpha=0.5, bins=20, 
                   label=f"Class {i}")
        
        ax.set_xlabel("Posterior Probability")
        ax.set_ylabel("Frequency")
        ax.set_title("Posterior Probability Distribution by Class")
        ax.legend()
        
        self.canvas.draw()
    
    def plot_pca_results(self, transformed_data, pca_model):
        """Plot PCA results"""
        self.figure.clear()
        
        # Plot first two components
        ax1 = self.figure.add_subplot(121)
        
        if self.y_test is not None and len(self.y_test) == transformed_data.shape[0]:
            scatter = ax1.scatter(transformed_data[:, 0], transformed_data[:, 1], 
                                c=self.y_test, cmap='viridis')
            self.figure.colorbar(scatter, ax=ax1)
        else:
            ax1.scatter(transformed_data[:, 0], transformed_data[:, 1])
            
        ax1.set_xlabel("Component 1")
        ax1.set_ylabel("Component 2")
        ax1.set_title("PCA: First Two Components")
        
        # Plot explained variance
        ax2 = self.figure.add_subplot(122)
        n_components = len(pca_model.explained_variance_ratio_)
        
        ax2.bar(range(1, n_components + 1), pca_model.explained_variance_ratio_)
        ax2.set_xlabel("Principal Component")
        ax2.set_ylabel("Explained Variance Ratio")
        ax2.set_title("Explained Variance by Component")
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def update_metrics(self, y_pred):
        """Update metrics display"""
        metrics_text = "Model Performance Metrics:\n\n"
        
        # Calculate appropriate metrics based on problem type
        if len(np.unique(self.y_test)) > 10:  # Regression
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            metrics_text += f"Mean Squared Error: {mse:.4f}\n"
            metrics_text += f"Root Mean Squared Error: {rmse:.4f}\n"
            metrics_text += f"Mean Absolute Error: {mae:.4f}\n"
            metrics_text += f"R² Score: {r2:.4f}"
            
        else:  # Classification
            accuracy = accuracy_score(self.y_test, y_pred)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            
            metrics_text += f"Accuracy: {accuracy:.4f}\n\n"
            
            # Compute F1 score for multi-class
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            metrics_text += f"F1 Score (weighted): {f1:.4f}\n\n"
            
            metrics_text += "Confusion Matrix:\n"
            metrics_text += str(conf_matrix)
            
            # Try to compute ROC and AUC for binary classification
            if len(np.unique(self.y_test)) == 2:
                try:
                    # Get probability estimates if the model supports predict_proba
                    if hasattr(self.current_model, "predict_proba"):
                        y_prob = self.current_model.predict_proba(self.X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
                        roc_auc = auc(fpr, tpr)
                        metrics_text += f"\n\nROC AUC: {roc_auc:.4f}"
                except:
                    # Some models don't support probability estimates
                    pass
        
        self.metrics_text.setText(metrics_text)
    
    def plot_training_history(self, history):
        """Plot neural network training history"""
        self.figure.clear()
        
        # Plot training & validation accuracy if available
        if 'accuracy' in history.history:
            ax1 = self.figure.add_subplot(211)
            ax1.plot(history.history['accuracy'])
            ax1.plot(history.history['val_accuracy'])
            ax1.set_title('Model Accuracy')
            ax1.set_ylabel('Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.legend(['Train', 'Test'])
            
            # Plot training & validation loss
            ax2 = self.figure.add_subplot(212)
            ax2.plot(history.history['loss'])
            ax2.plot(history.history['val_loss'])
            ax2.set_title('Model Loss')
            ax2.set_ylabel('Loss')
            ax2.set_xlabel('Epoch')
            ax2.legend(['Train', 'Test'])
        else:
            # Only plot loss if accuracy isn't available
            ax = self.figure.add_subplot(111)
            ax.plot(history.history['loss'])
            ax.plot(history.history['val_loss'])
            ax.set_title('Model Loss')
            ax.set_ylabel('Loss')
            ax.set_xlabel('Epoch')
            ax.legend(['Train', 'Test'])
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)

def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()