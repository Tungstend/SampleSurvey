import pandas as pd

def calculate_monthly_avg_depth(input_excel, output_excel=None):
    # 读取数据并保留前6列（时间 + 5口井）
    df = pd.read_excel(input_excel)
    df = df.iloc[:, :6]
    df.columns = ['Date', 'Well1', 'Well2', 'Well3', 'Well4', 'Well5']

    # 转换日期
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['YearMonth'] = df['Date'].dt.to_period('M')

    # 定义：只有该月非空值 ≥10 才计算均值
    def safe_avg(series):
        return series.mean() if series.count() >= 10 else pd.NA

    # 按年月分组，计算每口井的月均值
    monthly_avg = df.groupby('YearMonth')[['Well1', 'Well2', 'Well3', 'Well4', 'Well5']].agg(safe_avg)
    monthly_avg.index = monthly_avg.index.astype(str)

    # 可选输出
    if output_excel:
        monthly_avg.to_excel(output_excel)

    return monthly_avg

if __name__ == "__main__":
    input = "C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\zzzz\\国家级监测孔水位埋深变化.xlsx"
    output = "C:\\Users\\hanji\\Desktop\\tmp\\湖泊面积\\zzzz\\国家级监测孔水位埋深变化_月.xlsx"
    result = calculate_monthly_avg_depth(input, output)