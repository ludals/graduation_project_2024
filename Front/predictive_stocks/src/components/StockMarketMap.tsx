import React from "react";
import { Treemap, Tooltip, ResponsiveContainer } from "recharts";
import { proportionalSizes } from "./predefinedSizes";  // 비율 크기를 import

function getColor(change) {
  return change > 0 ? "#FF0000" : "#0000FF";
}

function parseCompanyName(name) {
  const parts = name.split("_");
  return parts[1];
}

export default function StockMarketMap({ stockData, isPrediction }) {
  // stockData를 시가총액 비율 크기 순으로 정렬
  const sortedData = stockData
    .map((stock) => ({
      name: parseCompanyName(stock.name),
      size: proportionalSizes[stock.name],  // 비율 크기 사용, 기본값 1
      change: stock.change,
      fill: getColor(stock.change),
    }))
    .sort((a, b) => b.size - a.size); // 큰 순서대로 정렬

  return (
    <div style={{ width: "100%", height: 600 }}>
      <h1>{isPrediction ? "Predicted Stock Market Map" : "Actual Stock Market Map"}</h1>
      <ResponsiveContainer>
        <Treemap
          data={sortedData}  // 정렬된 데이터를 Treemap에 전달
          dataKey="size"
          nameKey="name"
          stroke="#fff"
          fill="#8884d8"
          content={<CustomizedContent />}
          
        >
          <Tooltip />
        </Treemap>
      </ResponsiveContainer>
    </div>
  );
}

const CustomizedContent = (props) => {
  const { x, y, width, height, name, change } = props;
  const formattedChange = typeof change === 'number' ? change.toFixed(2) : '0.00';

  return (
    <g>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        style={{
          fill: getColor(change),
          stroke: "#fff",
          strokeWidth: 2,
          strokeOpacity: 1,
        }}
      />
      {width > 60 && height > 30 && (
        <>
          <text
            x={x + width / 2}
            y={y + height / 2 - 10}
            fill="#fff"
            fontSize={7} 
            textAnchor="middle"
            dominantBaseline="middle"
          >
            {name}
          </text>
          <text
            x={x + width / 2}
            y={y + height / 2 + 10}
            fill="#fff"
            fontSize={12}
            textAnchor="middle"
            dominantBaseline="middle"
          >
            {change > 0 ? `+${formattedChange}%` : `${formattedChange}%`}
          </text>
        </>
      )}
    </g>
  );
};
