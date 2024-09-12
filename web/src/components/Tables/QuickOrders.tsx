"use client";
import {
  Button,
  FormControl,
  FormLabel,
  Input,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  useDisclosure,
} from "@chakra-ui/react";
import axios from "axios";
import mongoose from "mongoose";
import Image from "next/image";
import { useEffect, useRef, useState } from "react";

const QuickOrders = () => {
  const [allQuickOrders, setAllQuickOrders]: any = useState([]);
  const [orderData, setOrderData] = useState({
    itemname: "",
    quantity: "",
    quickdelivery: true,
    expecteddelivery: new Date(),
  });
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [onSubmit, setOnSubmit] = useState(false);

  const { isOpen, onOpen, onClose } = useDisclosure();
  const initialRef = useRef(null);
  const finalRef = useRef(null);

  useEffect(() => {
    const fetchallOrders = async () => {
      const response = await axios.get("/api/products/orders/quickorders");
      setAllQuickOrders(response.data.quickorders);
    };
    fetchallOrders();
    setOnSubmit(false);
  }, [onSubmit]);

  const handleDelete = async (id: mongoose.Schema.Types.ObjectId) => {
    try {
      const response = await axios.delete(
        `/api/products/orders/deleteorder/${id}`,
      );
      setOnSubmit(true);
    } catch (error) {
      console.error("Error deleting the order:", error);
    }
  };

  const handleOrder = async () => {
    try {
      if (currentId) {
        await axios.patch(`/api/products/orders/modifyorder`, {
          ...orderData,
          _id: currentId,
        });
      } else {
        await axios.post("/api/products/orders/neworder", orderData);
      }
      setOnSubmit(true);
      onClose();
    } catch (error) {
      console.error("Error saving the order:", error);
    }
  };

  const handleEdit = (order: any) => {
    setOrderData({
      itemname: order.itemname,
      quantity: order.quantity,
      quickdelivery: order.quickdelivery,
      expecteddelivery: new Date(order.expecteddelivery),
    });
    setCurrentId(order._id);
    console.log(order);
    onOpen();
  };

  return (
    <div className="rounded-[10px] border border-stroke bg-white p-4 shadow-1 dark:border-dark-3 dark:bg-gray-dark dark:shadow-card sm:p-7.5">
      <Modal
        initialFocusRef={initialRef}
        finalFocusRef={finalRef}
        isOpen={isOpen}
        onClose={onClose}
      >
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>
            {currentId ? "Edit Order" : "Add a New Order"}
          </ModalHeader>
          <ModalCloseButton />
          <ModalBody pb={6}>
            <FormControl>
              <FormLabel>Items Name</FormLabel>
              <Input
                ref={initialRef}
                placeholder="Items Name"
                value={orderData.itemname}
                onChange={(e) =>
                  setOrderData({ ...orderData, itemname: e.target.value })
                }
              />
            </FormControl>
            <FormControl mt={4}>
              <FormLabel>Quantity</FormLabel>
              <Input
                type="number"
                placeholder="Quantity"
                value={orderData.quantity}
                onChange={(e) =>
                  setOrderData({ ...orderData, quantity: e.target.value })
                }
              />
            </FormControl>
            <FormControl mt={4}>
              <FormLabel>Suitable Delivery Date</FormLabel>
              <Input
                placeholder="Suitable Delivery"
                type="datetime-local"
                value={orderData.expecteddelivery.toISOString().slice(0, -1)}
                onChange={(e) =>
                  setOrderData({
                    ...orderData,
                    expecteddelivery: new Date(e.target.value),
                  })
                }
              />
            </FormControl>
            <FormControl mt={4}>
              <FormLabel>Quick Delivery (Write Yes)</FormLabel>
              <Input
                placeholder="Quick Delivery"
                value={orderData.quickdelivery ? "Yes" : "No"}
                onChange={(e) =>
                  setOrderData({
                    ...orderData,
                    quickdelivery: e.target.value.toLowerCase() === "yes",
                  })
                }
              />
            </FormControl>
          </ModalBody>

          <ModalFooter>
            <Button colorScheme="blue" mr={3} onClick={handleOrder}>
              {currentId ? "Update Order" : "Place Order"}
            </Button>
            <Button onClick={onClose}>Cancel</Button>
          </ModalFooter>
        </ModalContent>
      </Modal>

      <div className="max-w-full overflow-x-auto">
        <table className="w-full table-auto">
          <thead>
            <tr className="bg-[#F7F9FC] text-left dark:bg-dark-2">
              <th className="min-w-[220px] px-4 py-4 font-medium text-dark dark:text-white xl:pl-7.5">
                Urgent Items
              </th>
              <th className="min-w-[150px] px-4 py-4 font-medium text-dark dark:text-white">
                Quantity
              </th>
              <th className="min-w-[150px] px-4 py-4 font-medium text-dark dark:text-white">
                Order Date
              </th>
              <th className="min-w-[150px] px-4 py-4 font-medium text-dark dark:text-white">
                Expected Delivery
              </th>
              <th className="min-w-[120px] px-4 py-4 font-medium text-dark dark:text-white">
                Status
              </th>
              <th className="px-4 py-4 text-right font-medium text-dark dark:text-white xl:pr-7.5">
                Actions
              </th>
            </tr>
          </thead>
          <tbody>
            {allQuickOrders.map((item: any, index: any) => (
              <tr key={index}>
                <td
                  className={`border-[#eee] px-4 py-4 dark:border-dark-3 xl:pl-7.5 ${index === allQuickOrders.length - 1 ? "border-b-0" : "border-b"}`}
                >
                  <h5 className="text-dark dark:text-white">{item.itemname}</h5>
                </td>
                <td
                  className={`border-[#eee] px-4 py-4 dark:border-dark-3 ${index === allQuickOrders.length - 1 ? "border-b-0" : "border-b"}`}
                >
                  <p className="text-dark dark:text-white">{item.quantity}</p>
                </td>
                <td
                  className={`border-[#eee] px-4 py-4 dark:border-dark-3 ${index === allQuickOrders.length - 1 ? "border-b-0" : "border-b"}`}
                >
                  <p className="text-dark dark:text-white">
                    {new Date(item.orderplaced).toLocaleDateString("en-CA")}
                  </p>
                </td>
                <td
                  className={`border-[#eee] px-4 py-4 dark:border-dark-3 ${index === allQuickOrders.length - 1 ? "border-b-0" : "border-b"}`}
                >
                  <p className="text-dark dark:text-white">
                    {new Date(item.expecteddelivery).toLocaleDateString(
                      "en-CA",
                    )}
                  </p>
                </td>
                <td
                  className={`border-[#eee] px-4 py-4 dark:border-dark-3 ${index === allQuickOrders.length - 1 ? "border-b-0" : "border-b"}`}
                >
                  <p
                    className={`inline-flex rounded-full px-3.5 py-1 text-body-sm font-medium ${
                      item.deliverystatus === "Moderate"
                        ? "bg-[#219653]/[0.08] text-[#219653]"
                        : item.deliverystatus === "High"
                          ? "bg-[#D34053]/[0.08] text-[#D34053]"
                          : "bg-[#FFA70B]/[0.08] text-[#FFA70B]"
                    }`}
                  >
                    {item.deliverystatus}
                  </p>
                </td>
                <td
                  className={`border-[#eee] px-4 py-4 dark:border-dark-3 xl:pr-7.5 ${index === allQuickOrders.length - 1 ? "border-b-0" : "border-b"}`}
                >
                  <div className="flex items-center justify-end space-x-3.5">
                    <button
                      className="font-bold text-dark dark:text-white"
                      onClick={() => handleEdit(item)}
                    >
                      <Image
                        src="/vectors/edit.svg"
                        height={25}
                        width={25}
                        alt="Edit"
                        className="hover:rounded-full hover:shadow-md hover:shadow-gray-400"
                      />
                    </button>
                    <button className="font-bold text-dark dark:text-white">
                      <Image
                        src="/vectors/cross.svg"
                        height={25}
                        width={25}
                        alt="Delete"
                        className="p-1 hover:rounded-full hover:shadow-md hover:shadow-gray-400"
                        onClick={() => handleDelete(item._id)}
                      />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default QuickOrders;
