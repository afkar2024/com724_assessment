// src/api/axios.js
import axios from "axios";
import { toast } from "react-toastify";

// Create an Axios instance with baseURL
const api = axios.create({
    baseURL: "http://localhost:5000",
    timeout: 10000,
});

// Response interceptor for success and error handling
api.interceptors.response.use(
    (response) => {
        // Optional: toast on success for certain status codes
        // if (response.config.method !== 'get') {
        //   toast.success('Request successful')
        // }
        return response;
    },
    (error) => {
        const msg =
            error.response?.data?.error || error.message || "Unknown API error";
        toast.error(`API Error: ${msg}`);
        return Promise.reject(error);
    }
);
api.interceptors.request.use((config) => {
    console.log("[API REQUEST]", config.method, config.baseURL + config.url);
    return config;
});

export default api;
