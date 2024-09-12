import connect from "@/db/dbconnect";
import { getData } from "@/helpers/jwtToIdExtraction";
import Inventory from "@/models/inventoryModel";
import { NextRequest, NextResponse } from "next/server";

connect();

export async function PATCH(request: NextRequest) {
  try {
    // Get the user ID from the JWT token in the cookies
    const userId = getData(request);

    // Parse the request body to get the updated product details
    const reqBody = await request.json();
    const { itemname, inventorystock, visiblestock } = reqBody;

    // Extract the product ID from the URL (dynamic routing)
    const url = new URL(request.url);
    const id = url.pathname.split("/").pop(); // Extract ID from the URL

    if (!id) {
      return NextResponse.json(
        { error: "Invalid ID", success: false },
        { status: 400 },
      );
    }

    // Find the product by ID and ensure it belongs to the authenticated user
    const updatedProduct = await Inventory.findOneAndUpdate(
      { _id: id, user: userId }, // Ensure the product belongs to the authenticated user
      {
        itemname,
        inventorystock,
        visiblestock,
      },
      { new: true }, // This option returns the updated document
    );

    if (!updatedProduct) {
      return NextResponse.json(
        { error: "Product not found or unauthorized", success: false },
        { status: 404 },
      );
    }

    return NextResponse.json({
      message: "Product updated successfully",
      product: updatedProduct,
      success: true,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message, success: false },
      { status: 500 },
    );
  }
}
