import mongoose from "mongoose";

const demandSchema = new mongoose.Schema({
  productName: {
    type: String,
    required: true,
  },
  visibleStock: {
    type: Number,
    required: true,
  },
  demandStatus: {
    type: String,
    enum: ["High", "Moderate", "Low"],
    default: "Low",
    required: true,
  },
  demandFrequency: {
    type: Number,
    required: true,
  },
});

// Custom validator for demandFrequency
// demandSchema.path("demandFrequency").validate((value: any) => {
//   const frequency = parseFloat(value.toString());
//   return frequency >= 1.0 && frequency <= 10.0;
// }, "Demand frequency must be between 1.0 and 10.0");

const Demand =
  mongoose.models.demands || mongoose.model("demands", demandSchema);

export default Demand;
