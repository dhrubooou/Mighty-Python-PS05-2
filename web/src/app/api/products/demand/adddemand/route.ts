import connect from "@/db/dbconnect";
import Demand from "@/models/demandModel";
import { NextRequest, NextResponse } from "next/server";
import { getData } from "@/helpers/jwtToIdExtraction"; // Ensure this function is properly implemented

connect();

export async function POST(request: NextRequest) {
  try {
    // Extract user ID from the request using getData
    const userId = await getData(request);

    if (!userId) {
      return NextResponse.json(
        { error: "User not authenticated", success: false },
        { status: 401 },
      );
    }

    // Parse the request body
    const reqBody = await request.json();
    const { productName, visibleStock, demandStatus, demandFrequency } =
      reqBody;

    // Validate request body
    if (
      !productName ||
      visibleStock == null ||
      !demandStatus ||
      demandFrequency == null
    ) {
      return NextResponse.json(
        { error: "Missing required fields", success: false },
        { status: 400 },
      );
    }

    // Create and save a new demand
    const newDemand = new Demand({
      productName,
      visibleStock,
      demandStatus,
      demandFrequency,
    });

    await newDemand.save();

    return NextResponse.json({
      message: "Demand added successfully",
      demand: newDemand,
      success: true,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message, success: false },
      { status: 500 },
    );
  }
}
