import connect from "@/db/dbconnect";
import { getData } from "@/helpers/jwtToIdExtraction";
import Demand from "@/models/demandModel";
import { NextRequest, NextResponse } from "next/server";

connect();

export async function PATCH(request: NextRequest) {
  try {
    // Extract user ID from the request using getData
    const userId = await getData(request);

    if (!userId) {
      return NextResponse.json(
        { error: "User not authenticated", success: false },
        { status: 401 },
      );
    }

    // Extract the ID from the URL
    const url = new URL(request.url);
    const id = url.pathname.split("/").pop(); // Extract ID from the URL

    if (!id) {
      return NextResponse.json(
        { error: "Invalid ID", success: false },
        { status: 400 },
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

    // Find and update the demand
    const updatedDemand = await Demand.findByIdAndUpdate(
      id,
      { productName, visibleStock, demandStatus, demandFrequency },
      { new: true }, // This option returns the updated document
    );

    if (!updatedDemand) {
      return NextResponse.json(
        { error: "Demand not found", success: false },
        { status: 404 },
      );
    }

    return NextResponse.json({
      message: "Demand updated successfully",
      demand: updatedDemand,
      success: true,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message, success: false },
      { status: 500 },
    );
  }
}
