// Placeholder for Show2D widget
// TODO: Implement 2D image viewer widget

import * as React from "react";
import { createRender } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

function Show2D() {
  return (
    <Box sx={{ p: 2, border: "1px solid #444", borderRadius: 1 }}>
      <Typography>Show2D - Coming Soon</Typography>
    </Box>
  );
}

export const render = createRender(Show2D);
