import connect from "@/db/dbconnect";
import { getData } from "@/helpers/jwtToIdExtraction";
import Demand from "@/models/demandModel";
import { NextRequest, NextResponse } from "next/server";

connect();

export async function GET(request: NextRequest) {
  try {
    // Extract user ID from the request using getData
    const userId = await getData(request);

    if (!userId) {
      return NextResponse.json(
        { error: "User not authenticated", success: false },
        { status: 401 },
      );
    }

    // Fetch all demand data
    const demands = await Demand.find();

    return NextResponse.json({
      message: "Demand data retrieved successfully",
      demands,
      success: true,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message, success: false },
      { status: 500 },
    );
  }
}
