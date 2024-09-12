interface Demand {
  _id?: string;
  productName: string;
  visibleStock: number;
  demandStatus: "Low" | "Moderate" | "High";
  demandFrequency: number;
}
