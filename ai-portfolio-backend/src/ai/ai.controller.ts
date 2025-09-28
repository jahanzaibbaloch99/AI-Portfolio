import { Controller, Get, Query } from '@nestjs/common';
import { ApiTags, ApiQuery, ApiResponse } from '@nestjs/swagger';
import { AiService } from './ai.service';

@ApiTags('AI')
@Controller('ai')
export class AiController {
  constructor(private readonly aiService: AiService) {}

  @Get('ask')
  @ApiQuery({ name: 'question', required: true, description: 'Your question for AI' })
  @ApiResponse({ status: 200, description: 'AI Answer returned successfully' })
  async ask(@Query('question') question: string) {
    return this.aiService.askAI(question);
  }
}
