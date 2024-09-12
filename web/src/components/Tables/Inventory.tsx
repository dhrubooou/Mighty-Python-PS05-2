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
import React, { useEffect, useState } from "react";

const Inventory = () => {
  const [inventoryData, setInventoryData] = useState({
    itemname: "",
    inventorystock: "",
    visiblestock: "",
    addedate: new Date(),
  });
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [onSubmit, setOnSubmit] = useState(false);
  const [allProducts, setAllProducts]: any = useState([]);

  const { isOpen, onOpen, onClose } = useDisclosure();

  useEffect(() => {
    const fetchAllProducts = async () => {
      const response = await axios.get("/api/products/inventory/getallproduct");
      setAllProducts(response.data.inventorydata);
    };
    fetchAllProducts();
    setOnSubmit(false);
  }, [onSubmit]);

  const handleInventoryOperation = async () => {
    try {
      if (currentId) {
        // Update existing product
        await axios.patch(
          `/api/products/inventory/modifyinventory/${currentId}`,
          inventoryData,
        );
      } else {
        // Add new product
        await axios.post("/api/products/inventory/addproduct", inventoryData);
      }
      onClose();
      setOnSubmit(true);
    } catch (error) {
      console.error("Error handling inventory operation:", error);
    }
  };

  const handleDelete = async (id: mongoose.Schema.Types.ObjectId) => {
    try {
      await axios.delete(`/api/products/inventory/deleteproduct/${id}`);
      setOnSubmit(true);
    } catch (error) {
      console.error("Error deleting the Product:", error);
    }
  };

  const handleEdit = (product: any) => {
    setInventoryData({
      itemname: product.itemname,
      inventorystock: product.inventorystock,
      visiblestock: product.visiblestock,
      addedate: product.addedate,
    });
    setCurrentId(product._id);
    onOpen();
  };

  const initialRef = React.useRef(null);
  const finalRef = React.useRef(null);

  return (
    <div className="rounded-[10px] border border-stroke bg-white p-4 shadow-1 dark:border-dark-3 dark:bg-gray-dark dark:shadow-card sm:p-7.5">
      <button
        onClick={() => {
          setCurrentId(null); // Reset current ID for adding new product
          setInventoryData({
            itemname: "",
            inventorystock: "",
            visiblestock: "",
            addedate: new Date(),
          }); // Reset form
          onOpen();
        }}
        className="text-body-2xl mb-4 rounded-lg bg-blue-500 px-4 py-2 font-bold text-white dark:text-dark"
      >
        New Product
      </button>
      <Modal
        initialFocusRef={initialRef}
        finalFocusRef={finalRef}
        isOpen={isOpen}
        onClose={onClose}
      >
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>
            {currentId ? "Edit Product" : "Add a New Product"}
          </ModalHeader>
          <ModalCloseButton />
          <ModalBody pb={6}>
            <FormControl>
              <FormLabel>Items Name</FormLabel>
              <Input
                ref={initialRef}
                placeholder="Items Name"
                value={inventoryData.itemname}
                onChange={(e) =>
                  setInventoryData({
                    ...inventoryData,
                    itemname: e.target.value,
                  })
                }
              />
            </FormControl>
            <FormControl mt={4}>
              <FormLabel>Inventory Stock</FormLabel>
              <Input
                type="number"
                placeholder="Inventory Stock"
                value={inventoryData.inventorystock}
                onChange={(e) =>
                  setInventoryData({
                    ...inventoryData,
                    inventorystock: e.target.value,
                  })
                }
              />
            </FormControl>
            <FormControl mt={4}>
              <FormLabel>Visible Stock</FormLabel>
              <Input
                type="number"
                placeholder="Visible Stock"
                value={inventoryData.visiblestock}
                onChange={(e) =>
                  setInventoryData({
                    ...inventoryData,
                    visiblestock: e.target.value,
                  })
                }
              />
            </FormControl>
          </ModalBody>

          <ModalFooter>
            <Button
              colorScheme="blue"
              mr={3}
              onClick={handleInventoryOperation}
            >
              {currentId ? "Update Product" : "Add Product"}
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
                Product Name
              </th>
              <th className="min-w-[150px] px-4 py-4 font-medium text-dark dark:text-white">
                Stock Added Date
              </th>
              <th className="min-w-[150px] px-4 py-4 font-medium text-dark dark:text-white">
                Inventory Stock
              </th>
              <th className="min-w-[120px] px-4 py-4 font-medium text-dark dark:text-white">
                Visible Stock
              </th>
              <th className="px-4 py-4 text-right font-medium text-dark dark:text-white xl:pr-7.5">
                Actions
              </th>
            </tr>
          </thead>
          <tbody>
            {allProducts.map((product: any, index: any) => (
              <tr key={index}>
                <td
                  className={`border-[#eee] px-4 py-4 dark:border-dark-3 xl:pl-7.5 ${
                    index === allProducts.length - 1 ? "border-b-0" : "border-b"
                  }`}
                >
                  <h5 className="text-dark dark:text-white">
                    {product.itemname}
                  </h5>
                </td>
                <td
                  className={`border-[#eee] px-4 py-4 dark:border-dark-3 ${index === allProducts.length - 1 ? "border-b-0" : "border-b"}`}
                >
                  <p className="text-dark dark:text-white">
                    {new Date(product.addedate).toLocaleDateString("en-CA")}
                  </p>
                </td>
                <td
                  className={`border-[#eee] px-4 py-4 dark:border-dark-3 ${
                    index === allProducts.length - 1 ? "border-b-0" : "border-b"
                  }`}
                >
                  <p className="text-dark dark:text-white">
                    {product.inventorystock}
                  </p>
                </td>
                <td
                  className={`border-[#eee] px-4 py-4 dark:border-dark-3 ${
                    index === allProducts.length - 1 ? "border-b-0" : "border-b"
                  }`}
                >
                  {product.visiblestock}
                </td>
                <td
                  className={`border-[#eee] px-4 py-4 dark:border-dark-3 xl:pr-7.5 ${
                    index === allProducts.length - 1 ? "border-b-0" : "border-b"
                  }`}
                >
                  <div className="flex items-center justify-end space-x-3.5">
                    <button
                      className="font-bold text-dark dark:text-white"
                      onClick={() => handleEdit(product)}
                    >
                      <Image
                        src="/vectors/edit.svg"
                        height={25}
                        width={25}
                        alt="Edit"
                        className="hover:rounded-full hover:shadow-md hover:shadow-gray-400"
                      />
                    </button>
                    <button
                      className="font-bold text-dark dark:text-white"
                      onClick={() => handleDelete(product._id)}
                    >
                      <Image
                        src="/vectors/cross.svg"
                        height={25}
                        width={25}
                        alt="Delete"
                        className="p-1 hover:rounded-full hover:shadow-md hover:shadow-gray-400"
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

export default Inventory;
