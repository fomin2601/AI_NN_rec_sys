import { createSelector } from "@reduxjs/toolkit";

const inputItem = (state) => state?.check?.inputItem || {};

export const getCheckInputItem = createSelector(
  [inputItem], (inputItem) => inputItem
)