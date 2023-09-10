import { createSelector } from "@reduxjs/toolkit";

const recItem = (state) => state?.check?.recItem || {};

export const getCheckRecItem = createSelector(
  [recItem], (recItem) => recItem
)