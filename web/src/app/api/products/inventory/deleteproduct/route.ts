import { NextRequest, NextResponse } from "next/server";

export async function DELETE(request: NextRequest) {
  return NextResponse.json("Please Specify a right Product Id", {
    status: 500,
  });
}
