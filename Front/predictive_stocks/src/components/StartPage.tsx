import { useState, useEffect } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

export default function StartPage() {
  const [selectedCompany, setSelectedCompany] = useState("");
  const [chartData, setChartData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [correctedPrediction, setCorrectedPrediction] = useState(null);
  const companies = [
    "005930_daily_data",
    "Hyundai Motors",
    "SK Hynix",
    "LG Chem",
    // 추가적인 대기업 리스트
  ];

  useEffect(() => {
    if (selectedCompany) {
      // 회사가 선택되면 CSV 파일 로드
      loadCSVData(`/data/${selectedCompany}.csv`);
    }
  }, [selectedCompany]);

  const loadCSVData = async (filePath) => {
    const response = await fetch(filePath);
    const data = await response.text();

    const parsedData = parseCSVData(data);
    setChartData(parsedData);
  };

  const parseCSVData = (data) => {
    const rows = data.trim().split("\n");
    const headers = rows[0].split(",");

    // 데이터를 파싱한 후, 역순으로 정렬하여 가장 최근의 데이터가 마지막으로 오도록 함
    return rows.slice(1).map((row) => {
      const values = row.split(",");
      return {
        date: values[0],
        close: parseInt(values[1], 10),
        open: parseInt(values[2], 10),
        low: parseInt(values[3], 10),
        high: parseInt(values[4], 10),
        volume: parseInt(values[5], 10),
      };
    }).reverse(); // 데이터를 역순으로 정렬
};

  const handleCompanyChange = (event) => {
    setSelectedCompany(event.target.value);
  };

  const handlePredict = async () => {
    if (chartData) {
      const historicalClose = chartData.map((dataPoint) => dataPoint.close);
      const historicalVolume = chartData.map((dataPoint) => dataPoint.volume);

      // FastAPI 백엔드로 예측 요청
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          historical_close: historicalClose,
          historical_volume: historicalVolume,
        }),
      });

      const data = await response.json();
      setPrediction(data.predicted_close);
      setCorrectedPrediction(data.corrected_close);
    }
  };

  const lineChartData = {
    labels: chartData ? chartData.map((dataPoint) => dataPoint.date) : [],
    datasets: [
      {
        label: `${selectedCompany} 주가`,
        data: chartData ? chartData.map((dataPoint) => dataPoint.close) : [],
        fill: false,
        backgroundColor: "rgb(75, 192, 192)",
        borderColor: "rgba(75, 192, 192, 0.2)",
      },
    ],
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>AI 주가 예측 사이트</h1>

      <div style={{ marginBottom: "20px" }}>
        <label htmlFor="company-select">회사 선택:</label>
        <select
          id="company-select"
          value={selectedCompany}
          onChange={handleCompanyChange}
          style={{ marginLeft: "10px", padding: "5px" }}
        >
          <option value="">회사를 선택하세요</option>
          {companies.map((company) => (
            <option key={company} value={company}>
              {company}
            </option>
          ))}
        </select>
      </div>

      <div style={{ marginBottom: "20px" }}>
        {chartData ? (
          <div>
            <h2>{selectedCompany}의 주가 차트</h2>
            <Line data={lineChartData} />
          </div>
        ) : (
          <p>회사를 선택하면 차트가 표시됩니다.</p>
        )}
      </div>

      <div style={{ marginBottom: "20px" }}>
        <button onClick={handlePredict} disabled={!selectedCompany}>
          예측하기
        </button>
      </div>

      {prediction && (
        <div>
          <h2>예측 결과</h2>
          <p>단방향 예측된 주가: {prediction} 원</p>
          <p>양방향 수정된 주가: {correctedPrediction} 원</p>
        </div>
      )}
    </div>
  );
}
