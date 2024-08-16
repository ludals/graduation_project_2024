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

  const companies = [
    "삼성전자",
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

    return rows.slice(1).map((row) => {
      const values = row.split(",");
      const obj = {};
      headers.forEach((header, index) => {
        obj[header] = values[index];
      });
      return {
        date: obj["DateTime"],
        price: parseInt(obj["Close"].replace(/[\+\-]/g, ""), 10),
      };
    });
  };

  const handleCompanyChange = (event) => {
    setSelectedCompany(event.target.value);
  };

  const handlePredict = () => {
    // 더미 예측 결과
    const dummyPrediction = "81500";
    setPrediction(dummyPrediction);
  };

  const lineChartData = {
    labels: chartData ? chartData.map((dataPoint) => dataPoint.date) : [],
    datasets: [
      {
        label: `${selectedCompany} 주가`,
        data: chartData ? chartData.map((dataPoint) => dataPoint.price) : [],
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
          <p>다음 날의 주가: {prediction} 원</p>
        </div>
      )}
    </div>
  );
}
