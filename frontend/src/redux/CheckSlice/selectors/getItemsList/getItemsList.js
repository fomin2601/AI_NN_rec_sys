import { createSelector } from "@reduxjs/toolkit";

const itemsList = (state) => state?.check?.itemsList || [];

export const getCheckItemsList = createSelector(
  [itemsList], (itemsList) => itemsList
)