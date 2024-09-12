import connect from "@/db/dbconnect";
import { getData } from "@/helpers/jwtToIdExtraction";
import Orders from "@/models/orderModel";
import { NextRequest, NextResponse } from "next/server";

connect();

export async function PATCH(request: NextRequest) {
  try {
    // Get the user ID from the JWT token in the cookies
    const userId = getData(request);

    // Parse the request body to get the updated order details
    const reqBody = await request.json();
    const {
      itemname,
      quantity,
      expecteddelivery,
      quickdelivery,
      deliverystatus,
    } = reqBody;

    // Extract the order ID from the URL (dynamic routing)
    const url = new URL(request.url);
    const id = url.pathname.split("/").pop(); // Extract ID from the URL

    if (!id) {
      return NextResponse.json(
        { error: "Invalid ID", success: false },
        { status: 400 },
      );
    }

    // Find the order by ID and ensure the user ID matches before updating fields
    const updatedOrder = await Orders.findOneAndUpdate(
      { _id: id, user: userId }, // Ensure the order belongs to the authenticated user
      {
        itemname,
        quantity,
        expecteddelivery,
        quickdelivery,
        deliverystatus,
      },
      { new: true }, // This option returns the updated document
    );

    if (!updatedOrder) {
      return NextResponse.json(
        { error: "Order not found or unauthorized", success: false },
        { status: 404 },
      );
    }

    return NextResponse.json({
      message: "Order updated successfully",
      order: updatedOrder,
      success: true,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message, success: false },
      { status: 500 },
    );
  }
}
