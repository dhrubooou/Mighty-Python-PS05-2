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
} from "@chakra-ui/react";
import axios from "axios";
import Image from "next/image";
import { useEffect, useState } from "react";

const DemandTable = () => {
  const [demandData, setDemandData] = useState<Demand[]>([]);
  const [isAddModalOpen, setAddModalOpen] = useState(false);
  const [isEditModalOpen, setEditModalOpen] = useState(false);
  const [editDemand, setEditDemand] = useState<Demand | null>(null);
  const [newDemand, setNewDemand] = useState<Demand>({
    productName: "",
    visibleStock: 0,
    demandStatus: "Low",
    demandFrequency: 0,
  });

  useEffect(() => {
    // Fetch demand data from API
    const fetchDemandData = async () => {
      try {
        const response = await axios.get("/api/products/demand/getdemand");
        setDemandData(response.data.demands);
      } catch (error) {
        console.error("Error fetching demand data:", error);
      }
    };

    fetchDemandData();
  }, []);

  // Sort demand data by frequency
  const sortedDemandData = [...demandData].sort(
    (a, b) => b.demandFrequency - a.demandFrequency,
  );

  const handleOpenAddModal = () => setAddModalOpen(true);
  const handleCloseAddModal = () => setAddModalOpen(false);

  const handleOpenEditModal = (demand: Demand) => {
    setEditDemand(demand);
    setEditModalOpen(true);
  };

  const handleCloseEditModal = () => setEditModalOpen(false);

  const handleAddDemand = async () => {
    try {
      await axios.post("/api/products/demand/adddemand", newDemand);
      // Fetch updated demand data
      const response = await axios.get("/api/products/demand/getdemand");
      setDemandData(response.data.demands);
      handleCloseAddModal();
    } catch (error) {
      console.error("Error adding demand:", error);
    }
  };

  const handleEditDemand = async () => {
    if (!editDemand?._id) return; // Ensure _id exists
    try {
      await axios.patch(
        `/api/products/demand/modifydemand/${editDemand._id}`,
        editDemand,
      );
      // Fetch updated demand data
      const response = await axios.get("/api/products/demand/getdemand");
      setDemandData(response.data.demands);
      handleCloseEditModal();
    } catch (error) {
      console.error("Error editing demand:", error);
    }
  };

  return (
    <div className="rounded-[10px] border border-stroke bg-white p-4 shadow-1 dark:border-dark-3 dark:bg-gray-dark dark:shadow-card sm:p-7.5">
      <div className="mb-4">
        <Button colorScheme="blue" onClick={handleOpenAddModal}>
          Add Demand
        </Button>
      </div>
      <div className="max-w-full overflow-x-auto">
        <table className="w-full table-auto">
          <thead>
            <tr className="bg-[#F7F9FC] text-left dark:bg-dark-2">
              <th className="min-w-[220px] px-4 py-4 font-medium text-dark dark:text-white xl:pl-7.5">
                Demand Items
              </th>
              <th className="min-w-[150px] px-4 py-4 font-medium text-dark dark:text-white">
                Visible Stock
              </th>
              <th className="min-w-[120px] px-4 py-4 font-medium text-dark dark:text-white">
                Status
              </th>
              <th className="px-4 py-4 text-center font-medium text-dark dark:text-white xl:pr-7.5">
                Demand Frequency
              </th>
              <th className="px-4 py-4 text-center font-medium text-dark dark:text-white xl:pr-7.5">
                Actions
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedDemandData.map((demandItem) => (
              <tr key={demandItem._id || demandItem.productName}>
                <td
                  className={`border-[#eee] px-4 py-4 dark:border-dark-3 xl:pl-7.5`}
                >
                  <h5 className="text-dark dark:text-white">
                    {demandItem.productName}
                  </h5>
                </td>
                <td className={`border-[#eee] px-4 py-4 dark:border-dark-3`}>
                  <p className="text-dark dark:text-white">
                    {demandItem.visibleStock}
                  </p>
                </td>
                <td className={`border-[#eee] px-4 py-4 dark:border-dark-3`}>
                  <p
                    className={`inline-flex rounded-full px-3.5 py-1 text-body-sm font-medium ${
                      demandItem.demandStatus === "Moderate"
                        ? "bg-[#219653]/[0.08] text-[#219653]"
                        : demandItem.demandStatus === "High"
                          ? "bg-[#D34053]/[0.08] text-[#D34053]"
                          : "bg-[#FFA70B]/[0.08] text-[#FFA70B]"
                    }`}
                  >
                    {demandItem.demandStatus}
                  </p>
                </td>
                <td
                  className={`border-[#eee] px-4 py-4 dark:border-dark-3 xl:pr-7.5`}
                >
                  <div className="flex items-center justify-center space-x-3.5">
                    <p className="text-dark dark:text-white">
                      {demandItem.demandFrequency}
                    </p>
                  </div>
                </td>
                <td
                  className={`border-[#eee] px-4 py-4 dark:border-dark-3 xl:pr-7.5`}
                >
                  <div className="flex items-center justify-center space-x-3.5">
                    <button onClick={() => handleOpenEditModal(demandItem)}>
                      <Image
                        src="/vectors/edit.svg"
                        height={25}
                        width={25}
                        alt="Edit"
                        className="hover:rounded-xl hover:shadow-md hover:shadow-gray-400"
                      />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Add Demand Modal */}
      <Modal isOpen={isAddModalOpen} onClose={handleCloseAddModal}>
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Add Demand</ModalHeader>
          <ModalCloseButton />
          <ModalBody pb={6}>
            <FormControl>
              <FormLabel>Product Name</FormLabel>
              <Input
                placeholder="Product Name"
                value={newDemand.productName}
                onChange={(e) =>
                  setNewDemand({ ...newDemand, productName: e.target.value })
                }
              />
            </FormControl>
            <FormControl mt={4}>
              <FormLabel>Visible Stock</FormLabel>
              <Input
                type="number"
                placeholder="Visible Stock"
                value={newDemand.visibleStock}
                onChange={(e) =>
                  setNewDemand({
                    ...newDemand,
                    visibleStock: Number(e.target.value),
                  })
                }
              />
            </FormControl>
            <FormControl mt={4}>
              <FormLabel>Demand Status</FormLabel>
              <Input
                placeholder="Demand Status"
                value={newDemand.demandStatus}
                onChange={(e) =>
                  setNewDemand({
                    ...newDemand,
                    demandStatus: e.target.value as "Low" | "Moderate" | "High",
                  })
                }
              />
            </FormControl>
            <FormControl mt={4}>
              <FormLabel>Demand Frequency</FormLabel>
              <Input
                type="number"
                step="0.1"
                min="1"
                max="10"
                placeholder="Demand Frequency"
                value={newDemand.demandFrequency}
                onChange={(e) =>
                  setNewDemand({
                    ...newDemand,
                    demandFrequency: Number(e.target.value),
                  })
                }
              />
            </FormControl>
          </ModalBody>
          <ModalFooter>
            <Button colorScheme="blue" mr={3} onClick={handleAddDemand}>
              Save
            </Button>
            <Button onClick={handleCloseAddModal}>Cancel</Button>
          </ModalFooter>
        </ModalContent>
      </Modal>

      {/* Edit Demand Modal */}
      <Modal isOpen={isEditModalOpen} onClose={handleCloseEditModal}>
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Edit Demand</ModalHeader>
          <ModalCloseButton />
          <ModalBody pb={6}>
            {editDemand && (
              <>
                <FormControl>
                  <FormLabel>Product Name</FormLabel>
                  <Input
                    placeholder="Product Name"
                    value={editDemand.productName}
                    onChange={(e) =>
                      setEditDemand({
                        ...editDemand,
                        productName: e.target.value,
                      })
                    }
                  />
                </FormControl>
                <FormControl mt={4}>
                  <FormLabel>Visible Stock</FormLabel>
                  <Input
                    type="number"
                    placeholder="Visible Stock"
                    value={editDemand.visibleStock}
                    onChange={(e) =>
                      setEditDemand({
                        ...editDemand,
                        visibleStock: Number(e.target.value),
                      })
                    }
                  />
                </FormControl>
                <FormControl mt={4}>
                  <FormLabel>Demand Status</FormLabel>
                  <Input
                    placeholder="Demand Status"
                    value={editDemand.demandStatus}
                    onChange={(e) =>
                      setEditDemand({
                        ...editDemand,
                        demandStatus: e.target.value as
                          | "Low"
                          | "Moderate"
                          | "High",
                      })
                    }
                  />
                </FormControl>
                <FormControl mt={4}>
                  <FormLabel>Demand Frequency</FormLabel>
                  <Input
                    type="number"
                    step="0.1"
                    min="1"
                    max="10"
                    placeholder="Demand Frequency"
                    value={editDemand.demandFrequency}
                    onChange={(e) =>
                      setEditDemand({
                        ...editDemand,
                        demandFrequency: Number(e.target.value),
                      })
                    }
                  />
                </FormControl>
              </>
            )}
          </ModalBody>
          <ModalFooter>
            <Button colorScheme="blue" mr={3} onClick={handleEditDemand}>
              Save
            </Button>
            <Button onClick={handleCloseEditModal}>Cancel</Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </div>
  );
};

export default DemandTable;
