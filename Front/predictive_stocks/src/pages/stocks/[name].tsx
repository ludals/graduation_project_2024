import StockDetail from "@/components/StockDetail";
import { useRouter } from "next/router";

export default function StockPage() {
    const router = useRouter();
    const { name } = router.query;
  return <StockDetail name = {name} />;
}