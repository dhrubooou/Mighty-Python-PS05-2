import mongoose from "mongoose";

const inventorySchema = new mongoose.Schema({
  itemname: {
    type: String,
    required: true,
  },
  inventorystock: {
    type: Number,
    required: true,
    default: 0,
  },
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "users",
    required: true,
  },
  visiblestock: {
    type: Number,
    default: 0,
    required: true,
  },
  addedate: {
    type: Date,
    default: Date.now,
  },
});

const Inventory =
  mongoose.models.inventories || mongoose.model("inventories", inventorySchema);

export default Inventory;
