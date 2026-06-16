import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import {
  AnswerContent,
  prepareAnswerMarkdown,
} from "@/features/chat/answer-content";

describe("answer citations", () => {
  it("converts numbered citations without changing markdown links", () => {
    expect(prepareAnswerMarkdown("Kết quả [1] và [tài liệu](https://x.test)"))
      .toBe("Kết quả [1](citation:1) và [tài liệu](https://x.test)");
  });

  it("selects a matching evidence source", () => {
    const onCitationSelect = vi.fn();
    render(
      <AnswerContent
        answer="Nội dung được hỗ trợ bởi [1]."
        sourceCount={1}
        activeSourceIndex={-1}
        onCitationSelect={onCitationSelect}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Xem nguồn 1" }));
    expect(onCitationSelect).toHaveBeenCalledWith(0);
  });

  it("marks a citation as unavailable when its source is missing", () => {
    render(
      <AnswerContent
        answer="Nguồn bị thiếu [2]."
        sourceCount={1}
        activeSourceIndex={-1}
        onCitationSelect={() => undefined}
      />,
    );

    expect(
      screen.getByRole("button", { name: "Không tìm thấy nguồn 2" }),
    ).toBeDisabled();
  });
});
