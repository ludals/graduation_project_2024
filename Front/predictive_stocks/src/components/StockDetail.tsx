import { useRouter } from "next/router";
import { useState, useEffect } from "react";
import styled from "styled-components";
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

import { Companies } from "./companies";

export default function StockDetail({ name }) {
  const [chartData, setChartData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [correctedPrediction, setCorrectedPrediction] = useState(null);
  const [dLinearData, setDLinearData] = useState(null);
  const [lstmData, setLSTMData] = useState(null);
  const [transformerData, setTransformerData] = useState(null);

  useEffect(() => {
    if (name && Companies[name]) {
      fetchData(Companies[name]);
      fetchPredictions(Companies[name]); // 예측 데이터 가져오기
    }
  }, [name]);

  const fetchData = async (companyEnumValue) => {
    const response = await fetch(`/data/${companyEnumValue}.csv`);
    const data = await response.text();
    const parsedData = parseCSVData(data);
    setChartData(parsedData);
  };

  const fetchPredictions = async (companyEnumValue) => {
    const dLinearResponse = await fetch(
      `/data/predict_datas/DLinear_predictions/${name}_DLinear_test_predictions_close.csv`
    );
    const dLinearData = await parsePredictionData(await dLinearResponse.text());
    setDLinearData(dLinearData);

    const lstmResponse = await fetch(
      `/data/predict_datas/LSTM_predictions/${name}_LSTM_test_predictions_close.csv`
    );
    const lstmData = await parsePredictionData(await lstmResponse.text());
    setLSTMData(lstmData);

    const transformerResponse = await fetch(
      `/data/predict_datas/Transformer_predictions/${name}_Transformer_test_predictions_close.csv`
    );
    const transformerData = await parsePredictionData(await transformerResponse.text());
    setTransformerData(transformerData);
  };

  const parseCSVData = (data) => {
    const rows = data.trim().split("\n");
    return rows
      .slice(1)
      .reverse()
      .map((row) => {
        const values = row.split(",");
        return {
          date: values[0],
          open: parseInt(values[1], 10),
          high: parseInt(values[2], 10),
          low: parseInt(values[3], 10),
          close: parseInt(values[4], 10),
          volume: parseInt(values[5], 10),
        };
      });
  };

  const parsePredictionData = (data) => {
    const rows = data.trim().split("\n");
    return rows.slice(1).map((row) => {
      const [date, close] = row.split(",");
      return { date, close: parseFloat(close) };
    });
  };

  const synchronizeDataLength = (dataArrays) => {
    const minLength = Math.min(...dataArrays.map((arr) => arr.length));
    return dataArrays.map((arr) => arr.slice(-minLength));
  };

  useEffect(() => {
    if (chartData && dLinearData && lstmData && transformerData) {
      const [syncedChartData, syncedDLinear, syncedLSTM, syncedTransformer] =
        synchronizeDataLength([chartData, dLinearData, lstmData, transformerData]);

      setChartData(syncedChartData);
      setDLinearData(syncedDLinear);
      setLSTMData(syncedLSTM);
      setTransformerData(syncedTransformer);
    }
  }, [chartData, dLinearData, lstmData, transformerData]);

  const handlePredict = async () => {
    if (chartData) {
      const historicalClose = chartData.map((dataPoint) => dataPoint.close);
      const historicalVolume = chartData.map((dataPoint) => dataPoint.volume);

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
        label: `${name} 주가`,
        data: chartData ? chartData.map((dataPoint) => dataPoint.close) : [],
        fill: true,
        backgroundColor: "rgba(0, 0, 0, 0.1)",
        borderColor: "rgba(0, 0, 0, 1)",
        pointRadius: 0,
      },
      {
        label: "DLinear 예측",
        data: dLinearData ? dLinearData.map((point) => point.close) : [],
        borderColor: "rgba(255, 99, 132, 1)",
        borderWidth: 2,
        fill: false,
        pointRadius: 0,
      },
      {
        label: "LSTM 예측",
        data: lstmData ? lstmData.map((point) => point.close) : [],
        borderColor: "rgba(54, 162, 235, 1)",
        borderWidth: 2,
        fill: false,
        pointRadius: 0,
      },
      {
        label: "Transformer 예측",
        data: transformerData ? transformerData.map((point) => point.close) : [],
        borderColor: "rgba(75, 192, 192, 1)",
        borderWidth: 2,
        fill: false,
        pointRadius: 0,
      },
    ],
  };

  return (
    <div style={{ padding: "20px" }}>
      <div>
        {chartData ? (
          <div>
            <h2>{name}의 주가 차트</h2>
            <Line data={lineChartData} />
          </div>
        ) : (
          <p>데이터를 로드 중입니다...</p>
        )}
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
