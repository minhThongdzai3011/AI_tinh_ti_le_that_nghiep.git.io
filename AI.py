import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, layers, alpha=0.01):
        self.layers = layers
        self.alpha = alpha
        self.W = []
        self.b = []
        
        for i in range(len(layers) - 1):
            self.W.append(np.random.randn(layers[i], layers[i+1]) / np.sqrt(layers[i]))
            self.b.append(np.zeros((layers[i+1], 1)))

    def fit_partial(self, X, y):
        A = [X]
        for i in range(len(self.layers) - 1):
            A.append(sigmoid(np.dot(A[-1], self.W[i]) + self.b[i].T))

        y = y.reshape(-1, 1)
        dA = [A[-1] - y]
        dW = []
        db = []
        for i in reversed(range(len(self.layers) - 1)):
            dZ = dA[-1] * sigmoid_derivative(A[i + 1])
            dW_ = np.dot(A[i].T, dZ) / X.shape[0]
            db_ = np.sum(dZ, axis=0, keepdims=True).T / X.shape[0]
            dA_ = np.dot(dZ, self.W[i].T)
            dW.append(dW_)
            db.append(db_)
            dA.append(dA_)
        dW = dW[::-1]
        db = db[::-1]

        for i in range(len(self.layers) - 1):
            self.W[i] -= self.alpha * dW[i]
            self.b[i] -= self.alpha * db[i]

    def fit(self, X, y, epochs=20000, verbose=500):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        for epoch in range(epochs):
            self.fit_partial(X, y)
            if epoch % verbose == 0:
                loss = self.calculate_loss(X, y)
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        for i in range(len(self.layers) - 1):
            X = sigmoid(np.dot(X, self.W[i]) + self.b[i].T)
        return X

    def calculate_loss(self, X, y):
        y_predict = self.predict(X)
        y = y.reshape(-1, 1)
        loss = -np.mean(y * np.log(y_predict + 1e-12) + (1 - y) * np.log(1 - y_predict + 1e-12))
        return loss
    
def main():
    df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

    features = ['Age', 'TotalWorkingYears', 'Education', 'JobRole']
    df = df[features + ['Attrition']]

    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    df = pd.get_dummies(df, columns=['JobRole'], drop_first=True)

    X = df.drop(columns=['Attrition'])
    y = df['Attrition'].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = NeuralNetwork([X.shape[1], 10, 1], alpha=0.05)
    model.fit(X_train, y_train, epochs=20000, verbose=1000)

    tuoi = float(input("Nhập tuổi: "))
    kinh_nghiem = float(input("Nhập số năm kinh nghiệm: "))
    trinh_do_hoc_van = int(input("Nhập trình độ học vấn (1: Trung học, 2: Cao đẳng, 3: Đại học, 4: Sau đại học, 5: Tiến sĩ): "))
    nganh_nghe = input("Nhập ngành nghề (Sales, Research Scientist, Laboratory Technician, Manager, Healthcare Representative, Human Resources, Technical Degree, Manufacturing Director, Research Director): ")

    input_data = pd.DataFrame([[tuoi, kinh_nghiem, trinh_do_hoc_van, nganh_nghe]],
                              columns=['Age', 'TotalWorkingYears', 'Education', 'JobRole'])

    input_data = pd.get_dummies(input_data, columns=['JobRole'])

    for column in X.columns:
        if column not in input_data.columns:
            input_data[column] = 0
    
    input_data = input_data[X.columns]

    input_scaled = scaler.transform(input_data)

    ket_qua = model.predict(input_scaled)

    print(f' Xác suất thất nghiệp: {ket_qua[0][0] * 100:.2f}%')
    if ket_qua[0][0] > 0.5:
        print(' Có khả năng thất nghiệp')   
    else:   
        print(' Không có khả năng thất nghiệp')

if __name__ == "__main__":
    main()
