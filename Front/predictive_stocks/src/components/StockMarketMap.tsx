import React from "react";
import { Treemap, Tooltip, ResponsiveContainer } from "recharts";

const stockData = [
  { name: "AAPL", size: 2250, change: 1.5 },  // 상승
  { name: "MSFT", size: 2000, change: -0.7 }, // 하락
  { name: "GOOGL", size: 1500, change: 2.3 }, // 상승
  { name: "AMZN", size: 1700, change: -1.0 }, // 하락
  { name: "FB", size: 950, change: 3.0 },     // 상승
  { name: "TSLA", size: 800, change: -2.5 },  // 하락
  { name: "NVDA", size: 500, change: 1.8 },   // 상승
];

function getColor(change) {
  return change > 0 ? "#FF0000" : "#0000FF";
}

export default function StockMarketMap({isPrediction}) {
  const data = stockData.map((stock) => ({
    name: stock.name,
    size: stock.size,
    change: stock.change,
    fill: getColor(stock.change),
  }));

  return (
    <div style={{ width: "100%", height: 600 }}>
      <h1>{isPrediction ? "Predicted Stock Market Map" : "Actual Stock Market Map"}</h1>
      <ResponsiveContainer>
        <Treemap
          data={data}
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
  const { x, y, width, height, colors, name, change } = props;
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
          <text x={x + 10} y={y + 20} fill="#fff" fontSize={14}>
            {name}
          </text>
          <text x={x + 10} y={y + 40} fill="#fff" fontSize={12}>
            {change > 0 ? `+${change}%` : `${change}%`}
          </text>
        </>
      )}
    </g>
  );
};
