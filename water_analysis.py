import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import shap
import os


def compute_weighted_groundwater(df, gw_columns, distances):
    d_array = np.array([distances[col] for col in gw_columns])
    weights = 1 / (d_array + 1e-6)
    weights /= weights.sum()
    gw_data = df[gw_columns].values
    gw_weighted = np.dot(gw_data, weights)
    return gw_weighted


def create_sequences(X, y, time_steps=6):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def random_forest_shap(X, y, output_dir):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 原始SHAP图
    shap.summary_plot(shap_values, X, show=False)
    shap_path = os.path.join(output_dir, "shap_summary_plot.png")
    plt.tight_layout()
    plt.savefig(shap_path)
    plt.close()

    # 计算平均重要性得分
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    })
    importance_df.sort_values(by='mean_abs_shap', ascending=False, inplace=True)
    importance_df['normalized_score'] = importance_df['mean_abs_shap'] / importance_df['mean_abs_shap'].sum()

    # 保存为Excel
    excel_path = os.path.join(output_dir, "shap_feature_importance.xlsx")
    importance_df.to_excel(excel_path, index=False)

    # 条形图
    plt.figure(figsize=(8, 5))
    plt.barh(importance_df['feature'], importance_df['normalized_score'])
    plt.xlabel("Normalized SHAP Importance")
    plt.title("Feature Importance (SHAP - Random Forest)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "feature_importance_shap.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"[INFO] SHAP图像保存至：{shap_path}")
    print(f"[INFO] 权重图保存至：{fig_path}")
    print(f"[INFO] 特征权重表保存至：{excel_path}")

    return dict(zip(importance_df['feature'], importance_df['normalized_score']))


def train_lstm_model_fixed_12(X, y, time_steps):
    X_seq, y_seq = create_sequences(X, y, time_steps)
    print(f"[DEBUG] 构造时间序列样本数: {len(X_seq)}")

    if len(X_seq) < 12:
        raise ValueError(f"样本数不足12，实际仅有 {len(X_seq)} 个，请检查缺失数据或 time_steps 设置。")

    X_train = X_seq[:6]      # 用于预测第13~18月
    y_train = y_seq[:6]
    X_test = X_seq[6:12]     # 用于预测第19~24月
    y_test = y_seq[6:12]

    print(f"[DEBUG] X_train: {X_train.shape}, X_test: {X_test.shape}")

    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)

    return model, X_test, y_test


def main(
    excel_path: str,
    output_dir: str,
    target_col: str,
    feature_cols: list,
    gw_columns: list,
    distances: dict,
    time_steps: int
):
    df = pd.read_excel(excel_path)

    # 清洗地下水缺失值（插值填补）
    df[gw_columns] = df[gw_columns].interpolate(limit_direction='both')
    df['gw_weighted'] = compute_weighted_groundwater(df, gw_columns, distances)

    full_features = feature_cols + ['gw_weighted']
    X_raw = df[full_features].copy()
    y = df[target_col].copy()

    os.makedirs(output_dir, exist_ok=True)
    print("\n[INFO] 执行随机森林 + SHAP 分析...")
    shap_weight_dict = random_forest_shap(X_raw, y, output_dir)

    # 方案2：添加 SHAP 加权后的衍生特征
    X_augmented = X_raw.copy()
    for col in X_raw.columns:
        X_augmented[f"{col}_shap"] = X_raw[col] * shap_weight_dict[col]

    print("[INFO] 已生成 SHAP 加权衍生特征并拼接到原始输入中")

    # 归一化
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_augmented)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    print("\n[INFO] 开始训练 LSTM 模型...")
    model, X_test, y_test = train_lstm_model_fixed_12(X_scaled, y_scaled, time_steps=time_steps)

    y_pred = model.predict(X_test)
    print(f"[DEBUG] y_test shape: {y_test.shape}, y_pred shape: {y_pred.shape}")
    assert y_pred.shape == y_test.shape, "预测值与真实值数量不一致，检查样本划分逻辑"

    y_pred_inv = scaler_y.inverse_transform(y_pred)
    y_test_inv = scaler_y.inverse_transform(y_test)

    lstm_plot_path = os.path.join(output_dir, "lstm_prediction_plot.png")
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_inv, label='True')
    plt.plot(y_pred_inv, label='Predicted')
    plt.title("LSTM Water Level Prediction")
    plt.xlabel("Time Step")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(lstm_plot_path)
    plt.close()
    print(f"[INFO] LSTM预测图保存至：{lstm_plot_path}")

    result_df = pd.DataFrame({
        'True': y_test_inv.flatten(),
        'Predicted': y_pred_inv.flatten()
    })
    result_path = os.path.join(output_dir, "lstm_prediction_results.xlsx")
    result_df.to_excel(result_path, index=False)
    print(f"[INFO] 预测结果保存至：{result_path}")

# === 执行主程序 ===
if __name__ == "__main__":
    main(
        excel_path="C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\机器学习\\input_full.xlsx",
        output_dir="C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\机器学习\\结果",
        target_col="water_level",
        feature_cols=["temp", "rain", "evap", "ndvi_sum", "cropland_area"],
        gw_columns=["G1", "G2", "G3", "G4", "G5"],
        distances={"G1": 14281.830647, "G2": 25434.923713, "G3": 19483.333179, "G4": 41608.576516, "G5": 60767.629155},
        time_steps=12
    )