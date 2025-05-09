// src/components/SymbolDropdown.jsx
import React from "react";
import Select from "react-select";

// react-select styles overrides
const customStyles = {
    container: (base) => ({
        ...base,
        width: 240,
    }),
    control: (base, state) => ({
        ...base,
        backgroundColor: "white",
        borderColor: state.isFocused ? "#2684FF" : base.borderColor,
        boxShadow: state.isFocused ? "0 0 0 1px #2684FF" : base.boxShadow,
        "&:hover": { borderColor: "#2684FF" },
    }),
    menu: (base) => ({
        ...base,
        backgroundColor: "white",
        color: "black",
    }),
    menuPortal: (base) => ({
        ...base,
        zIndex: 9999,
    }),
    option: (base, state) => ({
        ...base,
        backgroundColor: state.isFocused ? "#f0f0f0" : "white",
        color: "black",
        "&:active": { backgroundColor: "#e6e6e6" },
    }),
    singleValue: (base) => ({
        ...base,
        color: "black",
    }),
};

export function SymbolDropdown({ symbols, value, onChange }) {
    const options = symbols.map((s) => ({ label: s, value: s }));

    return (
        <Select
            options={options}
            value={value ? { label: value, value } : null}
            onChange={(opt) => onChange(opt.value)}
            isSearchable
            placeholder="Select a symbol..."
            styles={customStyles}
            // portal the menu at the end of the document so it floats above everything
            menuPortalTarget={
                typeof document !== "undefined" ? document.body : null
            }
            menuPosition="fixed"
        />
    );
}
